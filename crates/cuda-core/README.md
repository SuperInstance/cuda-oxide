# cuda-core

Safe, idiomatic RAII wrappers around the CUDA driver API. This is the **entry point for GPU computing in Rust**: create a context, allocate memory, load kernels, launch them on streams, and move data between host and device — all with Rust's ownership and error-handling guarantees.

If you are writing a host application that drives the GPU, start here. If you are writing code that runs *on* the GPU, see [`cuda-device`](../cuda-device/). If you want `async`/`.await` integration, see [`cuda-async`](../cuda-async/).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [`CudaContext` — GPU Context Management](#cudacontext--gpu-context-management)
- [`CudaStream` — Async Command Queues](#cudastream--async-command-queues)
- [`CudaModule` / `CudaFunction` — Loading and Calling Kernels](#cudamodule--cudafunction--loading-and-calling-kernels)
- [`DeviceBuffer<T>` — Device Memory](#devicebuffert--device-memory)
- [`PinnedHostBuffer<T>` — Page-Locked Host Memory](#pinnedhostbuffert--page-locked-host-memory)
- [Raw Memory Operations (`memory` module)](#raw-memory-operations-memory-module)
- [Virtual Memory Management (`vmm` module)](#virtual-memory-management-vmm-module)
- [`CudaEvent` — Timing and Cross-Stream Synchronization](#cudaevent--timing-and-cross-stream-synchronization)
- [Peer-to-Peer Access (`peer` module)](#peer-to-peer-access-peer-module)
- [Error Handling Philosophy](#error-handling-philosophy)
- [Launch Configuration](#launch-configuration)
- [Raw Interop](#raw-interop)
- [Related Crates](#related-crates)

---

## Architecture Overview

The cuda-core design follows the CUDA driver's own resource hierarchy, adding RAII ownership at every level:

```text
CudaContext (Arc)           ← retains the device's primary context
    ├── CudaStream          ← non-blocking command queue
    │       ├── DeviceBuffer<T>   ← device allocation, freed on Drop
    │       ├── PinnedHostBuffer<T>  ← page-locked host staging area
    │       ├── CudaEvent     ← synchronization / timing primitive
    │       └── kernel launch ← enqueued work
    ├── CudaModule          ← PTX/cubin loaded into the context
    │       └── CudaFunction ← kernel entry point handle
    └── peer access         ← P2P memory mapping between contexts
```

Every wrapper holds an `Arc<CudaContext>` so the underlying CUDA context stays alive as long as any stream, buffer, event, module, or function handle exists. Dropping the last reference releases the primary context automatically.

**Context binding is automatic.** CUDA driver calls require a context to be "current" on the calling thread. Methods on cuda-core types call `bind_to_thread()` internally; you rarely need to manage `cuCtxSetCurrent` manually.

---

## Quick Start

```rust
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, launch_kernel_on_stream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a context on GPU 0
    let ctx = CudaContext::new(0)?;
    println!("Device: {}  (sm_{}.{})", ctx.device_name()?, ctx.compute_capability()?.0, ctx.compute_capability()?.1);

    // 2. Create a non-blocking stream
    let stream = ctx.new_stream()?;

    // 3. Allocate device memory
    let n = 1024usize;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let mut b_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    // 4. Load a kernel (compiled separately to PTX)
    let module = ctx.load_module_from_file("kernel.ptx")?;
    let func = module.load_function("scale")?;

    // 5. Launch
    let config = LaunchConfig::for_num_elems(n as u32);
    let mut params: &mut [*mut std::ffi::c_void] = &mut [
        &mut a_dev.cu_deviceptr() as *mut _ as *mut _,
        &mut b_dev.cu_deviceptr() as *mut _ as *mut _,
        &mut (n as u32) as *mut _ as *mut _,
    ];
    unsafe {
        launch_kernel_on_stream(
            &func,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            &stream,
            &mut params,
        )?;
    }

    // 6. Copy results back
    let b_host = b_dev.to_host_vec(&stream)?;
    println!("First result: {}", b_host[0]);

    Ok(())
}
```

---

## `CudaContext` — GPU Context Management

[`CudaContext`](src/context.rs) is the root of every CUDA operation. It retains the **primary context** for a given device ordinal via `cuDevicePrimaryCtxRetain` and releases it on `Drop`.

```rust
use cuda_core::CudaContext;

// GPU 0
let ctx = CudaContext::new(0)?;

// Context is bound to the calling thread automatically, but you can do it explicitly:
ctx.bind_to_thread()?;

// Query device properties
let name = ctx.device_name()?;               // e.g. "NVIDIA H100 80GB HBM3"
let (major, minor) = ctx.compute_capability()?; // e.g. (9, 0) for Hopper

// Synchronize all work in this context (blocks host thread)
ctx.synchronize()?;

// Sticky error tracking: if Drop paths record failures, they surface here
ctx.check_err()?;  // reads and clears the accumulated error state
```

The primary context is **shared across the process**: multiple `CudaContext::new(0)` calls return distinct `Arc<CudaContext>` handles that all reference the same underlying `CUcontext`. The context is released only when the last `Arc` drops.

**Thread safety:** `CudaContext` is `Send + Sync`. It uses atomics to track live stream counts and a sticky error state.

---

## `CudaStream` — Async Command Queues

[`CudaStream`](src/stream.rs) wraps a `CUstream` created with `CU_STREAM_NON_BLOCKING`. Operations enqueued on the same stream execute in FIFO order; operations on different streams may overlap.

```rust
use cuda_core::{CudaContext, CudaStream};

let ctx = CudaContext::new(0)?;
let s1 = ctx.new_stream()?;
let s2 = ctx.new_stream()?;

// Fork/join parallelism
let s3 = s1.fork()?;   // s3 waits on all prior work in s1, then runs independently
s3.synchronize()?;     // block host until s3 finishes
s1.join(&s3)?;         // make s1 wait on s3 before continuing

// Events for fine-grained cross-stream ordering
let evt = s1.record_event(None)?;  // None = disable timing (cheaper)
s2.wait(&evt)?;                    // s2 will not start work before evt is reached

// Host callbacks bridge CUDA streams to Rust closures
s1.launch_host_function(|| {
    println!("GPU work on s1 is done!");
})?;
```

**Default stream:** `ctx.default_stream()` returns a handle to the legacy default stream (stream `0`). It implicitly synchronizes with all other blocking streams in the same context. Prefer explicit non-blocking streams for performance.

---

## `CudaModule` / `CudaFunction` — Loading and Calling Kernels

[`CudaModule`](src/module.rs) loads compiled GPU code. [`CudaFunction`](src/module.rs) extracts a kernel entry point by name.

```rust
use cuda_core::CudaContext;

let ctx = CudaContext::new(0)?;

// From PTX source (JIT compiled by the driver)
let ptx = include_str!("kernel.ptx");
let module = ctx.load_module_from_ptx_src(ptx)?;

// From a cubin/PTX file on disk
let module = ctx.load_module_from_file("kernel.ptx")?;

// From an in-memory image (PTX bytes, cubin, or fatbin)
let module = ctx.load_module_from_image(&image_bytes)?;

// Extract the kernel handle
let func = module.load_function("my_kernel")?;

// `func` holds an Arc to `module`, so the module cannot be unloaded while the
// function handle is live. Clone is cheap (Arc bump).
```

**Device globals:** Resolve `__constant__` or `__device__` symbols and write to them from the host:

```rust
let (dptr, size) = module.get_global("MY_CONST")?;
// Use ConstantHandle for typed, stream-ordered writes
```

---

## `DeviceBuffer<T>` — Device Memory

[`DeviceBuffer<T>`](src/device_buffer.rs) is the device-side equivalent of `Vec<T>`. It owns a `CUdeviceptr` and frees it on `Drop` via `cuMemFree`. The buffer carries no implicit stream reference; the stream is an explicit parameter on every transfer.

```rust
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

// Allocate + H2D copy (async, enqueued on stream)
let host = vec![1.0f32, 2.0, 3.0];
let buf = DeviceBuffer::from_host(&stream, &host)?;

// Zero-initialized allocation
let zeros = DeviceBuffer::<f32>::zeroed(&stream, 1024)?;

// Blocking D2H copy (synchronizes stream internally)
let back = buf.to_host_vec(&stream)?;

// Non-blocking D2H into an existing slice (synchronizes)
let mut dst = vec![0.0f32; 3];
buf.copy_to_host(&stream, &mut dst)?;
```

**The `DeviceCopy` trait:** Only types that are safe to bitwise-copy between host and device can be stored in a `DeviceBuffer`. This includes all scalar numeric types, arrays of `DeviceCopy` types, tuples up to 8 elements, pointers, `f16`, and `bf16`. It explicitly excludes `String`, `Vec`, and other heap-owned types.

```rust
// This will not compile:
// let _ = DeviceBuffer::<String>::zeroed(&stream, 1);
```

---

## `PinnedHostBuffer<T>` — Page-Locked Host Memory

[`PinnedHostBuffer<T>`](src/pinned_host_buffer.rs) allocates page-locked ("pinned") host memory via `cuMemAllocHost`. Pinned memory is required for **true asynchronous** host-device transfers — without it, the driver must stage copies through an internal pinned buffer, which adds latency and prevents overlap.

```rust
use cuda_core::{CudaContext, PinnedHostBuffer, CudaStream, DeviceBuffer};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

// Allocate pinned memory and copy host data into it
let pinned = PinnedHostBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0])?;

// Async H2D from pinned memory (no implicit sync)
let dev = unsafe { DeviceBuffer::from_pinned_host(&stream, &pinned)? };

// Async D2H back into pinned memory (caller must sync later)
let mut pinned_out = PinnedHostBuffer::<f32>::zeroed(&ctx, 3)?;
unsafe { dev.copy_to_pinned_host_async(&stream, &mut pinned_out)? };
stream.synchronize()?;  // now safe to read pinned_out
```

**Safety note:** The `*_async` pinned variants are `unsafe` because they only enqueue the copy and return. The caller must ensure the pinned buffer is not dropped, freed, or aliased until the stream reaches a synchronization point.

---

## Raw Memory Operations (`memory` module)

For code that already holds raw `CUdeviceptr` values, the [`memory`](src/memory.rs) module provides the underlying primitives. All functions are `unsafe` because they operate on raw pointers.

| Operation | Async (stream-ordered) | Sync (blocking) |
|-----------|------------------------|-----------------|
| Allocate | `malloc_async` | `malloc_sync` |
| Free | `free_async` | `free_sync` |
| H2D copy | `memcpy_htod_async` | `memcpy_htod_sync` |
| D2H copy | `memcpy_dtoh_async` | — |
| D2D copy | `memcpy_dtod_async` | — |
| Memset | `memset_d8_async` | — |
| Pinned alloc | — | `malloc_host` |
| Pinned free | — | `free_host` |

Use the async variants when you want the operation to be ordered relative to other work on the same stream. Use the sync variants for one-off allocations where you do not yet have a stream.

---

## Virtual Memory Management (`vmm` module)

The [`vmm`](src/vmm.rs) module exposes the CUDA Virtual Memory Management API (sm_70+), which separates **physical allocation**, **virtual address reservation**, and **mapping** into independent steps. This is the foundation for advanced use cases such as peer-to-peer symmetric heaps and sparse memory.

```rust
use cuda_core::vmm::{PhysicalAllocation, VirtualReservation, Mapping, set_access, allocation_granularity};

let device = ctx.cu_device();
let gran = allocation_granularity(device)?;
let size = cuda_core::vmm::align_size(1024 * 1024, gran);

// 1. Allocate physical memory
let phys = PhysicalAllocation::new(device, size)?;

// 2. Reserve virtual address space
let va = VirtualReservation::new(size, 0)?;

// 3. Map physical memory into the VA range
let mapping = Mapping::new(va.base(), size, &phys, 0)?;

// 4. Grant access (required before the VA is usable)
set_access(va.base(), size, &[device])?;

// Drop order matters: mapping → then phys / va
```

---

## `CudaEvent` — Timing and Cross-Stream Synchronization

[`CudaEvent`](src/event.rs) is a lightweight synchronization primitive. Record it on one stream, then make another stream wait on it.

```rust
use cuda_core::CudaContext;

let ctx = CudaContext::new(0)?;
let s1 = ctx.new_stream()?;
let s2 = ctx.new_stream()?;

// Timing-enabled event (pass Some(CU_EVENT_DEFAULT))
let start = ctx.new_event(Some(cuda_core::sys::CU_EVENT_DEFAULT))?;
let end   = ctx.new_event(Some(cuda_core::sys::CU_EVENT_DEFAULT))?;

start.record(&s1)?;
// ... enqueue work on s1 ...
end.record(&s1)?;

let ms = start.elapsed_ms(&end)?;  // synchronizes both events internally

// Ordering without timing (cheaper)
let evt = s1.record_event(None)?;  // None = CU_EVENT_DISABLE_TIMING
s2.wait(&evt)?;
```

---

## Peer-to-Peer Access (`peer` module)

The [`peer`](src/peer.rs) module enables direct memory access between GPUs over NVLink or PCIe.

```rust
use cuda_core::{CudaContext, peer};

let ctx0 = CudaContext::new(0)?;
let ctx1 = CudaContext::new(1)?;

if peer::can_access_peer(&ctx0, &ctx1)? {
    peer::enable_peer_access(&ctx0, &ctx1)?;  // ctx0 can now read/write ctx1's memory
    // ... launch kernels that access peer memory ...
    peer::disable_peer_access(&ctx0, &ctx1)?;
}
```

---

## Error Handling Philosophy

Every fallible cuda-core operation returns `Result<T, DriverError>`. [`DriverError`](src/error.rs) wraps a raw `CUresult` code and implements `std::error::Error`, `Display`, and `Debug` by querying the driver for human-readable names and descriptions via `cuGetErrorName` / `cuGetErrorString`.

```rust
use cuda_core::DriverError;

match some_cuda_call() {
    Ok(v) => v,
    Err(e) => {
        // e.g. "DriverError(1, \"CUDA_ERROR_INVALID_VALUE\")"
        eprintln!("CUDA error: {}", e);
        // Access the raw code when you need to match on specific errors
        if e.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY {
            // handle OOM
        }
    }
}
```

**[`IntoResult`](src/error.rs)** converts raw `(CUresult, MaybeUninit<T>)` pairs into `Result<T, DriverError>`. On `CUDA_SUCCESS` the `MaybeUninit` is assumed initialized; on failure it is discarded safely. This pattern is used throughout the codebase to wrap every driver call.

**Sticky errors:** `CudaContext` tracks a sticky error state atomically. If a `Drop` path encounters a driver error (e.g. freeing a stream fails), it records the error rather than panicking. The next call to `bind_to_thread()` or `check_err()` surfaces it.

---

## Launch Configuration

[`LaunchConfig`](src/launch.rs) bundles grid dimensions, block dimensions, and dynamic shared memory size.

```rust
use cuda_core::LaunchConfig;

// 1-D launch for n elements with a block size of 256
let cfg = LaunchConfig::for_num_elems(1024);
// grid_dim  = (4, 1, 1)
// block_dim = (256, 1, 1)
// shared_mem_bytes = 0

// Custom configuration
let cfg = LaunchConfig {
    grid_dim: (grid_x, grid_y, 1),
    block_dim: (block_x, block_y, 1),
    shared_mem_bytes: 4096,  // 4 KiB dynamic shared memory
};
```

**Cluster launches (Hopper+):** Use [`launch_kernel_ex_on_stream`](src/lib.rs) to launch kernels with thread-block cluster dimensions.

**Cooperative launches:** Use [`launch_kernel_cooperative_on_stream`](src/lib.rs) for grid-wide barriers (`cuda_device::grid::sync()`). The driver guarantees all blocks are co-resident.

---

## Raw Interop

Sometimes you need to pass a raw CUDA handle to another library (e.g. cuBLAS, NCCL, NVSHMEM). Each wrapper exposes its underlying handle via an `unsafe` accessor:

```rust
// Module
let raw_module: cuda_core::sys::CUmodule = unsafe { module.cu_module() };

// Function
let raw_func: cuda_core::sys::CUfunction = unsafe { func.cu_function() };

// Stream
let raw_stream: cuda_core::sys::CUstream = stream.cu_stream();

// Context
let raw_ctx: cuda_core::sys::CUcontext = ctx.cu_ctx();
let raw_dev: cuda_core::sys::CUdevice = ctx.cu_device();
```

**Contract:** The returned handle is non-owning. You must keep the cuda-oxide wrapper alive for at least as long as the raw handle is used. Do not transfer or destroy the handle. Bind the owning context before making raw driver calls.

---

## Related Crates

| Crate | Role |
|-------|------|
| [`cuda-device`](../cuda-device/) | Write CUDA kernels in Rust (`#![no_std]` device-side intrinsics) |
| [`cuda-async`](../cuda-async/) | `async`/`.await` integration with CUDA streams |
| [`cuda-macros`](../cuda-macros/) | Proc-macros: `#[kernel]`, `#[device]`, `#[launch_bounds]` |
| [`cuda-host`](../cuda-host/) | Higher-level host abstractions built on cuda-core |
| [`cuda-bindings`](../cuda-bindings/) | Raw FFI to `cuda.h` (used internally by cuda-core) |

---

## License

Licensed under the [Apache-2.0](../../LICENSE-APACHE) license. Portions derived from NVIDIA CUDA samples are licensed under [LICENSE-NVIDIA](../../LICENSE-NVIDIA).
