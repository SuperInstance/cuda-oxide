# cuda-async

Async execution layer for CUDA device operations. This crate turns GPU work into
lazy, composable Rust futures. Kernels, memory copies, and arbitrary callbacks
are described first, scheduled later, and executed concurrently across a pool of
CUDA streams.

## Overview

```text
  module.kernel_async(...)
         |
         v
  AsyncKernelLaunch          <-- lazy description, no GPU work yet
         |
    .and_then(|()| ...)      <-- compose with other DeviceOperations
         |
    .sync() / .await         <-- SchedulingPolicy picks a stream, executes
         |
         v
  cuLaunchKernel(stream)     <-- actual GPU dispatch
  cuLaunchHostFunc(stream)   <-- host callback wakes the Rust future
```

## Core abstractions

### `DeviceOperation`

The trait at the center of the crate. A `DeviceOperation` is a lazy, stream-agnostic
unit of GPU work. It carries no CUDA stream affinity and performs no side effects
when constructed. Work is only submitted to the GPU when the operation is *executed*
inside an `ExecutionContext`, which binds it to a concrete stream.

| Method | Stream chosen by | Blocks thread? |
|--------|------------------|----------------|
| `.await` | Thread-local `SchedulingPolicy` | No (suspends async task) |
| `.sync()` | Thread-local `SchedulingPolicy` | Yes |
| `.sync_on(&stream)` | Caller-provided stream | Yes |
| `.async_on(&stream)` | Caller-provided stream | No |

**Combinators** build dataflow graphs without touching hardware:

| Combinator | Effect |
|------------|--------|
| `.and_then(f)` | Sequence: run `self`, then `f(result)` |
| `.and_then_with_context(f)` | Like `and_then`, closure sees `ExecutionContext` |
| `.apply(f)` | Alias for `and_then` |
| `.arc()` | Wrap output in `Arc<T>` for sharing |
| `zip!(a, b, c?)` | Run 2–3 operations, return tuple of results |
| `unzip!(op)` | Split a tuple-producing operation into independent branches |

### `DeviceFuture`

Bridges CUDA stream completion to Rust's `Future` trait. On the first poll:

1. The operation is executed on its assigned stream.
2. A host callback is registered via `cuLaunchHostFunc`.
3. When the GPU reaches the callback, it wakes the future through an `AtomicWaker`.

No busy-waiting. The executor parks the task until the stream signals completion.

### `SchedulingPolicy`

Decides which CUDA stream an operation runs on. Policies are `Sync` and shared
across all operations on a device context.

| Policy | Behaviour |
|--------|-----------|
| `StreamPoolRoundRobin` (default) | Rotates across *N* streams for automatic overlap of independent kernels and copies |
| `SingleStream` | Serialises all work onto one stream (debugging, strict ordering) |

### `DeviceBox<T>`

Owning smart pointer for device memory. On drop it enqueues `cuMemFreeAsync` on
a dedicated per-device deallocator stream rather than blocking with `cuMemFree`.
This eliminates the full-device synchronization that synchronous free would cause.

**Safety contract:** all streams that reference the allocation must be synchronized
before the box is dropped.

### `AsyncDeviceContext`

Thread-local per-device state maintained inside a `thread_local!`:

- Primary `CudaContext` for driver API calls.
- `SchedulingPolicy` for stream selection.
- Dedicated deallocator stream for `DeviceBox` drops.
- Kernel function cache keyed by `FunctionKey` hashes.

Initialized once per thread via `init_device_contexts(default_device_id, num_devices)`.

## Async streams and events

Streams are created during policy initialization (`SchedulingPolicy::init`) and
recycled for the lifetime of the thread. The round-robin pool (default size: 4)
automatically distributes work so that independent kernels execute concurrently
on the GPU when hardware resources permit.

```rust
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_module;
use cuda_core::LaunchConfig;

// 1. Initialize once per thread.
init_device_contexts(0, 1)?;
let module = kernels::load_async(0)?;

// 2. Build a lazy operation.
let op = module.vecadd_async(
    LaunchConfig::for_num_elems(1024),
    &a_dev,
    &b_dev,
    &mut c_dev,
)?;

// 3. Execute — policy picks the next stream in the pool.
op.sync()?;       // blocking
// or: op.await?  // async, suspends the task
```

## Concurrent kernel execution

Because `StreamPoolRoundRobin` assigns each scheduled operation to a different
stream (wrapping at the pool size), kernels that have no data dependencies
naturally overlap. The GPU scheduler interleaves their execution automatically.

```rust
let op_a = module.kernel_a_async(config, &buf_a, &mut out_a)?;
let op_b = module.kernel_b_async(config, &buf_b, &mut out_b)?;

// Run both concurrently on different streams, wait for both.
let (a, b) = zip!(op_a, op_b).sync()?;
```

## Async memory operations

`DeviceBox` is the async-friendly counterpart to `DeviceBuffer`. It wraps a raw
`CUdeviceptr` and frees asynchronously:

```rust
use cuda_async::device_box::DeviceBox;

let boxed: DeviceBox<[f32]> = unsafe {
    DeviceBox::from_raw_parts(dptr, len, device_id)
};

// boxed is passed to kernels as a slice argument...
// ...and when it goes out of scope, cuMemFreeAsync is enqueued
// on the deallocator stream with no host blocking.
```

## When to use async vs sync API

| Situation | Recommended API | Why |
|-----------|----------------|-----|
| Single kernel, fire-and-forget | `module.kernel(&stream, ...)` (sync API in `cuda-host`) | Simplest, least overhead |
| Chained kernels with data deps | `DeviceOperation::and_then` | Lazy composition, automatic stream selection |
| Multiple independent kernels | `zip!` + `.sync()` / `.await` | Automatic concurrency via stream pool |
| Spawned Tokio/async-std task | `module.kernel_async_owned(...)` | Operation owns buffers, returns them on completion |
| Long-lived pipeline / DAG | `DeviceOperation` combinators | Build once, schedule many times |
| Debugging ordering issues | `SingleStream` policy | Strict FIFO, no overlap |

**Borrowed vs owned async:**

- **Borrowed** (`kernel_async`): returns `AsyncKernelLaunch<'_>`. Rust borrows
  referenced buffers until the operation completes. Use in the current stack frame.

- **Owned** (`kernel_async_owned`): takes device buffers by value, keeps them
  alive for the GPU work, and returns them as the operation output. Use when the
  operation must leave the current scope (e.g., spawned in a Tokio task).

```rust
// Borrowed: buffers stay borrowed until .sync() returns.
module.vecadd_async(config, &a_dev, &b_dev, &mut c_dev)?.sync()?;

// Owned: buffers are moved into the operation and returned after completion.
let (a_dev, b_dev, c_dev) = module
    .vecadd_async_owned(config, a_dev, b_dev, c_dev)?
    .await?;
```

## Buffer lifetime safety

Async launches are lazy: building the operation does not enqueue GPU work. Raw
pointer launches are easy to misuse because a `CUdeviceptr` is just an integer
handle with no Rust lifetime attached.

```text
raw async launch:
  build operation from CUdeviceptr
  drop the owning buffer
  run operation later  -> kernel sees stale memory

typed borrowed async:
  module.kernel_async(..., &input, &mut output)
  Rust keeps those buffers borrowed until completion

typed owned async:
  module.kernel_async_owned(..., input, output)
  the operation owns the buffers and can be spawned safely
```

Prefer generated `#[cuda_module]` async methods for application code. Use raw
device pointers only when you can prove the allocation outlives every scheduled
operation that touches it.
