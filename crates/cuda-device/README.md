# cuda-device

`#![no_std]` device-side intrinsics and abstractions for writing CUDA kernels in Rust. This crate provides everything that executes **on the GPU**: thread identification, memory abstractions, synchronization primitives, warp-level collectives, tensor cores, TMA, atomics, and debug facilities.

If you are writing host code that drives the GPU, see [`cuda-core`](../cuda-core/). If you want `async`/`.await` integration, see [`cuda-async`](../cuda-async/).

## Table of Contents

- [Programming Model](#programming-model)
- [Thread Indexing](#thread-indexing)
- [`DisjointSlice<T>` — Safe Parallel Writes](#disjointslice-t--safe-parallel-writes)
- [Shared Memory](#shared-memory)
- [Warp-Level Primitives](#warp-level-primitives)
- [Cooperative Groups](#cooperative-groups)
- [Grid-Wide Synchronization](#grid-wide-synchronization)
- [Thread Block Clusters](#thread-block-clusters)
- [Atomics](#atomics)
- [Barriers (`mbarrier`)](#barriers-mbarrier)
- [Tensor Memory Accelerator (TMA)](#tensor-memory-accelerator-tma)
- [Tensor Cores — WGMMA (Hopper)](#tensor-cores--wgmma-hopper)
- [Tensor Cores — tcgen05 (Blackwell)](#tensor-cores--tcgen05-blackwell)
- [Debug Facilities](#debug-facilities)
- [Memory Fences](#memory-fences)
- [Proc-Macro Re-exports](#proc-macro-re-exports)
- [Safety Model](#safety-model)
- [Architecture Support Matrix](#architecture-support-matrix)

---

## Programming Model

A CUDA kernel is a function annotated with `#[kernel]` that is launched by the host with a grid of threads. Threads are organized hierarchically:

```text
Grid                          ← all blocks in the launch
└── Thread Block (CTA)        ← up to 1024 threads
    └── Warp                  ← 32 threads (hardware scheduling unit)
        └── Thread            ← single SIMD lane
```

On Hopper (sm_90+), an additional level exists:

```text
Grid
└── Thread Block Cluster      ← up to 8 blocks with DSMEM
    └── Thread Block
        └── Warp
            └── Thread
```

Kernels in this crate are compiled by `rustc-codegen-cuda`: Rust MIR → LLVM IR → PTX → cubin. The `cuda-device` crate supplies the intrinsic functions that lower to PTX instructions.

```text
  User kernel code
       │
       │  uses
       ▼
  cuda-device  ─────────────────────────────────┐
  │                                             │
  │  thread     warp       barrier     cusimd   │  Universal
  │  disjoint   shared     atomic      debug    │
  │  fence      grid       coop_grps            │
  │                                             │
  │  tma        wgmma      stmatrix    cluster  │  Hopper+
  │  tcgen05    clc                             │  Blackwell+
  └─────────────────────────────────────────────┘
       │
       │  compiled by
       ▼
  rustc-codegen-cuda  →  MIR → LLVM IR → PTX
```

---

## Thread Indexing

The [`thread`](src/thread.rs) module provides hardware intrinsics for querying a thread's position in the launch grid. These lower to PTX special register reads (`%tid.x`, `%ctaid.x`, etc.).

### Raw Intrinsics

```rust
use cuda_device::thread;

let tid_x  = thread::threadIdx_x();   // 0 .. blockDim.x-1
let bid_x  = thread::blockIdx_x();    // 0 .. gridDim.x-1
let bdim_x = thread::blockDim_x();    // threads per block, X dimension
let gdim_x = thread::gridDim_x();     // blocks per grid, X dimension

// Y and Z variants are also available:
// threadIdx_y, blockIdx_y, blockDim_y, gridDim_y
// threadIdx_z, blockIdx_z, blockDim_z, gridDim_z
```

### Safe Index Helpers

For the common case where each thread processes one element, use the typed index helpers. They return a [`ThreadIndex`](src/thread.rs) witness that guarantees uniqueness per thread, enabling safe parallel writes to [`DisjointSlice`](#disjointslice-t--safe-parallel-writes).

```rust
use cuda_device::{thread, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[i] + b[i];
    }
}
```

| Function | Returns | Use When |
|----------|---------|----------|
| `thread::index_1d()` | `ThreadIndex<Index1D>` | 1-D grid launches |
| `thread::index_2d::<S>()` | `Option<ThreadIndex<Index2D<S>>>` | 2-D grids with const row stride `S` |
| `unsafe thread::index_2d_runtime(s)` | `Option<ThreadIndex<Runtime2DIndex>>` | Runtime strides (caller asserts uniformity) |
| `thread::index_2d_row()` | `usize` | Row component only |
| `thread::index_2d_col()` | `usize` | Column component only |

**Why the witness type matters:** `ThreadIndex` is `!Send + !Sync + !Copy + !Clone` and scoped to the kernel body. It cannot be forged by user code, smuggled through shared memory, or outlive the kernel. `DisjointSlice<T, Index2D<128>>` will reject a `ThreadIndex<Index2D<256>>` at compile time, preventing stride mismatches.

### Block Synchronization

```rust
use cuda_device::thread;

// All threads in the block must reach this barrier
thread::sync_threads();  // PTX: bar.sync 0
```

**Deadlock hazard:** Never place `sync_threads()` inside a conditional where not all threads in the block take the same branch.

---

## `DisjointSlice<T>` — Safe Parallel Writes

[`DisjointSlice<T, IndexSpace>`](src/disjoint.rs) is a slice-like type that can only be accessed with a thread-unique `ThreadIndex`. This makes parallel writes data-race-free without explicit synchronization.

```rust
use cuda_device::{kernel, thread, DisjointSlice};

#[kernel]
pub fn saxpy(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, alpha: f32) {
    // One-shot form: mint index and resolve in a single call
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = alpha * a[i] + b[i];
    }
}
```

**API summary:**

| Method | Safety | Description |
|--------|--------|-------------|
| `get_mut(idx)` | safe | Bounds-checked `Option<&mut T>` |
| `get_mut_indexed()` | safe | Mint `ThreadIndex` + resolve in one call |
| `get_unchecked_mut(idx)` | `unsafe` | No bounds check; caller guarantees validity |

For non-trivial access patterns (reductions, histograms), `get_unchecked_mut` is the escape hatch when your algorithm already guarantees disjoint access.

---

## Shared Memory

Shared memory is a fast, block-scoped scratchpad (typically 164–228 KiB per SM). All threads in a block see the same shared memory; different blocks have independent copies.

### Static Shared Arrays (`SharedArray`)

[`SharedArray<T, N, ALIGN>`](src/shared.rs) declares compile-time-sized shared memory. It is intentionally `!Sync` because concurrent access requires GPU barriers that the Rust type system cannot see.

```rust
use cuda_device::{kernel, thread, SharedArray, DisjointSlice};

#[kernel]
pub fn tiled_matvec(a_row: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    unsafe {
        TILE[tid] = a_row[tid];
    }
    thread::sync_threads();

    // Now all threads can read what other threads wrote
    unsafe {
        let sum = (0..256).map(|i| TILE[i]).sum::<f32>();
        if tid == 0 {
            *out.get_unchecked_mut(thread::blockIdx_x() as usize) = sum;
        }
    }
}
```

**Alignment:** Use `ALIGN = 128` for TMA destinations:
```rust
static mut TMA_TILE: SharedArray<f32, 256, 128> = SharedArray::UNINIT;
```

### Dynamic Shared Arrays

[`DynamicSharedArray<T, ALIGN>`](src/shared.rs) uses runtime-sized shared memory configured at launch via `LaunchConfig::shared_mem_bytes`.

```rust
use cuda_device::{kernel, DynamicSharedArray, thread};

#[kernel]
pub fn flexible_kernel(data: &[f32]) {
    let smem: *mut f32 = DynamicSharedArray::<f32>::get();
    unsafe {
        *smem.add(thread::threadIdx_x() as usize) = data[thread::index_1d().get()];
    }
    thread::sync_threads();
    // ...
}

// Host-side launch:
// config.shared_mem_bytes = 2048;  // 512 f32s
```

---

## Warp-Level Primitives

A warp is 32 threads that execute in SIMT lockstep. The [`warp`](src/warp.rs) module exposes register-to-register operations that are much faster than shared memory (~2 cycles vs ~20 cycles) and require no explicit synchronization.

### Lane Identification

```rust
use cuda_device::warp;

let lane = warp::lane_id();   // 0-31
let warp = warp::warp_id();   // threadIdx.x / 32
```

### Shuffle Operations

Exchange data between lanes in the same warp without going through shared memory:

```rust
use cuda_device::warp;

// Broadcast lane 0's value to all lanes
let broadcasted = warp::shuffle(my_value, 0);

// Butterfly exchange (XOR by lane_mask)
let paired = warp::shuffle_xor(my_value, 1);  // swap with neighbor

// Read from lane (my_lane + delta)
let from_below = warp::shuffle_down(my_value, 1);

// Read from lane (my_lane - delta)
let from_above = warp::shuffle_up(my_value, 1);
```

`f32` variants are also available: `shuffle_f32`, `shuffle_xor_f32`, `shuffle_down_f32`, `shuffle_up_f32`.

### Warp Reduction (Manual)

```rust
use cuda_device::{kernel, thread, warp, DisjointSlice};

#[kernel]
pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
    let gid = thread::index_1d();
    let mut val = data[gid.get()];

    // Butterfly reduction
    val += warp::shuffle_xor_f32(val, 16);
    val += warp::shuffle_xor_f32(val, 8);
    val += warp::shuffle_xor_f32(val, 4);
    val += warp::shuffle_xor_f32(val, 2);
    val += warp::shuffle_xor_f32(val, 1);

    if warp::lane_id() == 0 {
        let warp_idx = gid.get() / 32;
        unsafe { *out.get_unchecked_mut(warp_idx) = val; }
    }
}
```

### Warp Vote Operations

```rust
use cuda_device::warp;

// True if ALL lanes have predicate true
let all_positive = warp::all(value > 0.0);

// True if ANY lane has predicate true
let any_nan = warp::any(value.is_nan());

// 32-bit mask: bit i set iff lane i's predicate is true
let mask = warp::ballot(value > 0.0);
let count = mask.count_ones();              // how many lanes match
let first_match = mask.trailing_zeros();    // lowest matching lane
```

### Match Operations (sm_70+)

Find lanes that hold the same value — useful for deduplication and leader election:

```rust
use cuda_device::warp;

// Bitmask of lanes whose value equals mine
let same_key = warp::match_any_sync(u32::MAX, key);
let leader = same_key.trailing_zeros();
if warp::lane_id() == leader {
    // Only one lane per unique key executes this
}

// Full mask if all lanes agree, else 0
let unanimous = warp::match_all_sync(u32::MAX, key) != 0;
```

### Masked Sync Variants

Inside divergent control flow, use the `*_sync(mask, ...)` variants with `warp::active_mask()`:

```rust
if some_predicate {
    let mask = warp::active_mask();
    // ... do divergent work ...
    warp::sync_mask(mask);  // convergence point
    let leader = mask.trailing_zeros();
    let value = warp::shuffle_sync(mask, my_value, leader);
}
```

---

## Cooperative Groups

The [`cooperative_groups`](src/cooperative_groups.rs) module wraps raw warp intrinsics in typed handles so the participation mask is part of the type system rather than a silent integer.

### The Universal Trio

Every group implements `ThreadGroup`:

```rust
use cuda_device::cooperative_groups::{this_thread_block, ThreadGroup};

let block = this_thread_block();
let n = block.size();           // threads per block
let rank = block.thread_rank(); // my index in the block
block.sync();                   // block-wide barrier
```

### Group Hierarchy

```rust
use cuda_device::cooperative_groups::{
    this_grid, this_cluster, this_thread_block,
    coalesced_threads, ThreadGroup, WarpCollective,
};

let grid = this_grid();              // entire launch (cooperative only)
let cluster = this_cluster();        // thread block cluster (sm_90+)
let block = this_thread_block();     // current CTA
let warp = block.tiled_partition::<32>();  // full warp
let half = block.tiled_partition::<16>();  // sub-warp tile
let active = coalesced_threads();    // currently-converged lanes
```

### Warp Collectives

`WarpTile<N>` and `CoalescedThreads` implement `WarpCollective`:

```rust
use cuda_device::cooperative_groups::{this_thread_block, WarpCollective};

let block = this_thread_block();
let warp = block.tiled_partition::<32>();

let m = warp.ballot(tag == HIT);
if m != 0 {
    let leader = m.trailing_zeros();
    let payload = warp.shfl(my_payload, leader);
}
```

Switching to a 16-lane tile is one line:
```rust
let tile = block.tiled_partition::<16>();
let m = tile.ballot(tag == HIT);  // mask isolates each half-warp
```

### Reductions and Scans

Built-in warp and block reductions using a typed `ReduceOp` trait:

```rust
use cuda_device::cooperative_groups::{
    this_thread_block, warp_reduce, block_reduce, ops::Sum,
};

let warp = this_thread_block().tiled_partition::<32>();
let total: u32 = warp_reduce::<u32, Sum, _>(&warp, my_value);
// Every lane receives the reduced value

// Block reduction needs shared memory scratch space
static mut SCRATCH: cuda_device::SharedArray<u32, 32> = cuda_device::SharedArray::UNINIT;
let block = this_thread_block();
let total: u32 = unsafe {
    block_reduce::<u32, Sum, _>(&block, my_value, &raw mut SCRATCH)
};
```

Supported operations: `Sum`, `Min`, `Max` for `u32`/`i32`/`f32`; `BitAnd`, `BitOr`, `BitXor` for `u32`.

---

## Grid-Wide Synchronization

The [`grid`](src/grid.rs) module provides `sync()`, a grid-wide barrier usable only in **cooperative kernel launches**. The host must launch with `cuda_core::launch_kernel_cooperative` or `cuda_launch! { cooperative: true, ... }`.

```rust
use cuda_device::{kernel, thread, grid, DisjointSlice};

#[kernel]
pub fn rehash(buckets: &mut [Bucket]) {
    let gid = thread::index_1d();

    // Phase 1: every thread reads its old slot
    let snapshot = read_bucket(buckets, gid.get());

    grid::sync();  // ALL blocks must reach here

    // Phase 2: safe to write because all reads completed globally
    write_to_new_slot(buckets, gid.get(), snapshot);
}
```

**Non-cooperative launches deadlock at `grid::sync()`** because the driver does not guarantee all blocks are co-resident and does not populate the grid workspace pointer.

---

## Thread Block Clusters

The [`cluster`](src/cluster.rs) module exposes Hopper's thread-block cluster API. Clusters group up to 8 blocks that can access each other's shared memory (Distributed Shared Memory, DSMEM) and synchronize at cluster granularity.

```rust
use cuda_device::{kernel, thread, cluster, SharedArray, DisjointSlice};

#[kernel]
#[cluster_launch(4, 1, 1)]  // 4 blocks per cluster
pub fn cluster_example(mut output: DisjointSlice<u32>) {
    static mut SHMEM: SharedArray<u32, 256> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x();
    let my_rank = cluster::block_rank();

    // Each block writes to its shared memory
    if tid == 0 {
        unsafe { SHMEM.as_mut_ptr().write(my_rank * 100) };
    }
    thread::sync_threads();
    cluster::cluster_sync();  // all blocks in cluster have written

    // Read neighbor's shared memory via DSMEM
    let neighbor = (my_rank + 1) % cluster::cluster_size();
    let neighbor_ptr = unsafe { cluster::map_shared_rank(SHMEM.as_ptr(), neighbor) };
    let value = unsafe { *neighbor_ptr };
}
```

---

## Atomics

Two atomic APIs coexist:

### `cuda_device::atomic` — Scoped GPU Atomics

Explicit scope control (`Device`, `Block`, `System`) across six value types (`u32`, `i32`, `u64`, `i64`, `f32`, `f64`):

```rust
use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

#[kernel]
pub fn count_hits(...) {
    COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
}
```

### `core::sync::atomic` — Standard Library Atomics

Standard `AtomicU32`, `AtomicBool`, etc. also compile to GPU code, defaulting to device scope. Both paths emit the same NVVM atomic ops.

---

## Barriers (`mbarrier`)

The [`barrier`](src/barrier.rs) module provides hardware async barriers (sm_80+) for tracking TMA and MMA completion without busy-waiting on `sync_threads()`.

### Raw Barrier API

```rust
use cuda_device::{kernel, thread, SharedArray};
use cuda_device::barrier::{Barrier, mbarrier_init, mbarrier_arrive, mbarrier_wait};

#[kernel]
pub fn async_kernel(...) {
    static mut BAR: Barrier = Barrier::UNINIT;

    if thread::threadIdx_x() == 0 {
        unsafe { mbarrier_init(&raw mut BAR, 128); }  // 128 arrivals expected
    }
    thread::sync_threads();

    let token = unsafe { mbarrier_arrive(&raw const BAR) };
    // ... do independent work ...
    unsafe { mbarrier_wait(&raw const BAR, token); }
}
```

### Typestate-Managed Barrier (`ManagedBarrier`)

Compile-time lifecycle tracking prevents use-before-init, double-init, and use-after-inval:

```rust
use cuda_device::barrier::{
    ManagedBarrier, Uninit, TmaBarrier,
    mbarrier_init, fence_proxy_async_shared_cta,
};

static mut BAR: cuda_device::barrier::Barrier = cuda_device::barrier::Barrier::UNINIT;

let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
let bar = unsafe { bar.init(128) };  // transitions to Ready
// bar.arrive(), bar.wait(token), bar.try_wait(token) ...
let _dead = unsafe { bar.inval() };  // transitions to Invalidated
```

**Critical for TMA:** Call `fence_proxy_async_shared_cta()` after `mbarrier_init` and before issuing TMA operations. This ensures the barrier initialization is visible to the async proxy hardware.

---

## Tensor Memory Accelerator (TMA)

TMA is a hardware DMA unit on Hopper+ that performs asynchronous bulk tensor copies without consuming thread resources. The [`tma`](src/tma.rs) module exposes the device-side copy instructions.

### Global → Shared (G2S)

```rust
use cuda_device::{kernel, thread, SharedArray, barrier::{Barrier, mbarrier_init}};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};

#[kernel]
pub fn tma_kernel(tensor_map: *const TmaDescriptor, ...) {
    static mut TILE: SharedArray<f32, 4096> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;

    if thread::threadIdx_x() == 0 {
        unsafe {
            mbarrier_init(&raw mut BAR, 1);
        }
    }
    thread::sync_threads();

    // Thread 0 initiates TMA copy
    if thread::threadIdx_x() == 0 {
        unsafe {
            cp_async_bulk_tensor_2d_g2s(
                &raw mut TILE as *mut u8,
                tensor_map,
                tile_x, tile_y,
                &raw mut BAR,
            );
        }
    }

    // All threads wait for TMA completion
    let token = unsafe { cuda_device::barrier::mbarrier_arrive(&raw const BAR) };
    unsafe { cuda_device::barrier::mbarrier_wait(&raw const BAR, token); }
}
```

### Multicast (Cluster-Wide G2S)

Deliver the same tile to multiple CTAs in a cluster simultaneously:

```rust
use cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast;

unsafe {
    cp_async_bulk_tensor_2d_g2s_multicast(
        dst, tensor_map, coord0, coord1, barrier, 0b1111,  // deliver to ranks 0-3
    );
}
```

### Shared → Global (S2G)

S2G uses `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group` instead of mbarrier:

```rust
use cuda_device::tma::{cp_async_bulk_tensor_2d_s2g, cp_async_bulk_commit_group, cp_async_bulk_wait_group};

unsafe {
    cp_async_bulk_tensor_2d_s2g(src, tensor_map, x, y);
}
cp_async_bulk_commit_group();
cp_async_bulk_wait_group(0);  // wait for all pending groups
```

### TMA Dimensions

G2S and S2G variants exist for 1D through 5D tensors: `cp_async_bulk_tensor_{1,2,3,4,5}d_g2s` and `cp_async_bulk_tensor_{1,2,3,4,5}d_s2g`.

---

## Tensor Cores — WGMMA (Hopper)

WGMMA (Warpgroup Matrix Multiply-Accumulate) operates at the warpgroup level (128 threads) on Hopper. The [`wgmma`](src/wgmma.rs) module exposes fence/commit/wait synchronization and MMA instructions.

```rust
use cuda_device::wgmma::*;

#[kernel]
pub fn wgmma_kernel(a_smem: *const u8, b_smem: *const u8, ...) {
    // Each thread holds 32 floats of the 64×64 accumulator
    let mut acc: Acc64x64 = zero_accumulator();

    // Create SMEM descriptors
    let desc_a = unsafe { make_smem_desc(a_smem) };
    let desc_b = unsafe { make_smem_desc(b_smem) };

    wgmma_fence();

    // Issue WGMMA (K=16 per call; loop for larger K)
    for k in 0..4 {
        let off = k * 16 * std::mem::size_of::<bf16>();
        unsafe {
            wgmma_mma_m64n64k16_f32_bf16(
                &mut acc,
                desc_a + off as u64,
                desc_b + off as u64,
            );
        }
    }

    wgmma_commit_group();
    wgmma_wait_group::<0>();

    // acc now holds the 64×64 result tile
}
```

**Supported formats:**
- `wgmma_mma_m64n64k16_f32_bf16` — bf16 inputs, f32 accumulate
- `wgmma_mma_m64n64k16_f32_f16` — f16 inputs, f32 accumulate
- `wgmma_mma_m64n64k16_f32_tf32` — tf32 inputs, f32 accumulate

---

## Tensor Cores — tcgen05 (Blackwell)

Blackwell introduces tcgen05, which replaces WGMMA with **single-thread MMA semantics** and a new memory type called **Tensor Memory (TMEM)**. The [`tcgen05`](src/tcgen05.rs) module manages TMEM allocation, MMA operations, and barrier integration.

### Key Differences from WGMMA

| Aspect | WGMMA (Hopper) | tcgen05 (Blackwell) |
|--------|---------------|---------------------|
| MMA issue | 128 threads collectively | **1 thread** |
| Matrix A/D storage | Registers/SMEM | **Tensor Memory (TMEM)** |
| Allocation | Implicit | **Dynamic (`tcgen05_alloc`)** |
| Wait mechanism | `wgmma_wait_group` | **`mbarrier_try_wait`** |

### Usage Pattern

```rust
use cuda_device::tcgen05::*;
use cuda_device::barrier::*;

#[kernel]
pub fn blackwell_mma(a_desc: u64, b_desc: u64, idesc: Tcgen05InstructionDescriptor, ...) {
    static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    static mut MBAR: Barrier = Barrier::UNINIT;

    // 1. Allocate TMEM (warp-synchronous)
    let tmem = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
    let tmem = unsafe { tmem.alloc() };  // All threads call; warp 0 allocates
    let addr = tmem.address();

    // 2. Initialize barrier
    if thread::threadIdx_x() == 0 {
        unsafe { mbarrier_init(&raw mut MBAR, 1); }
    }
    thread::sync_threads();

    // 3. Single thread issues MMA
    if thread::threadIdx_x() == 0 {
        tcgen05_fence_before_thread_sync();
        unsafe {
            tcgen05_mma_ws_f16(
                addr.raw(), addr.raw(),  // D and A in TMEM
                a_desc, b_desc, idesc.raw(),
                false,  // no D accumulator reuse
            );
        }
        unsafe { tcgen05_commit(&raw mut MBAR as *mut u64); }
    }

    // 4. All threads wait
    let token = unsafe { mbarrier_arrive(&raw const MBAR) };
    unsafe { mbarrier_wait(&raw const MBAR, token); }

    // 5. Deallocate TMEM (MUST happen before kernel exit)
    let _dead = unsafe { tmem.dealloc() };
}
```

**TMEM lifecycle:** All allocated TMEM **must** be deallocated before the kernel exits. Failure results in `CUDA_ERROR_TENSOR_MEMORY_LEAK`.

**Instruction descriptors:** Use the builder API for compile-time-safe configuration:

```rust
let idesc = Tcgen05InstructionDescriptor::builder()
    .shape(Tcgen05MmaShape::M128_N256)
    .element_type(Tcgen05ElementType::BF16)
    .accumulator_type(Tcgen05AccumulatorType::F32)
    .build();
```

---

## Debug Facilities

The [`debug`](src/debug.rs) module provides device-side debugging tools:

```rust
use cuda_device::{gpu_printf, gpu_assert, thread};

#[kernel]
pub fn debug_kernel(data: &[f32]) {
    let idx = thread::index_1d();
    gpu_printf!("thread %d: val = %f\n", idx.get() as i32, data[idx.get()] as f64);
    gpu_assert!(data[idx.get()] >= 0.0);
}
```

| Function | Description |
|----------|-------------|
| `gpu_printf!(format, ...)` | Device-side `vprintf` with C vararg promotion |
| `gpu_assert!(expr)` | Trap on failure |
| `debug::clock()` | Per-SM clock register (low overhead) |
| `debug::clock64()` | 64-bit clock |
| `debug::globaltimer()` | Global nanosecond timer |
| `debug::trap()` | Unconditional trap |
| `debug::breakpoint()` | Debugger breakpoint |

---

## Memory Fences

The [`fence`](src/fence.rs) module provides PTX memory fences for ordering visibility across scopes:

```rust
use cuda_device::fence;

fence::threadfence_block();   // block-scope ordering
fence::threadfence();         // device-scope ordering
fence::threadfence_system();  // system-scope ordering (includes host)
```

---

## Proc-Macro Re-exports

These attributes are defined in `cuda-macros` and re-exported from `cuda-device` for convenience:

| Attribute | Purpose |
|-----------|---------|
| `#[kernel]` | Mark a function as a GPU kernel entry point |
| `#[device]` | Mark a helper function or extern block for device compilation |
| `#[launch_bounds(max_threads, min_blocks)]` | Set `.maxntid` and `.minnctapersm` PTX directives |
| `#[cluster_launch(x, y, z)]` | Set compile-time cluster dimensions (`.reqnctapercluster`) |
| `#[convergent]` | Mark as convergent (barrier semantics) |
| `#[pure]` | Mark as pure (no side effects) |
| `#[readonly]` | Mark as read-only |
| `gpu_printf!` | Device-side printf macro |

---

## Safety Model

1. **`ThreadIndex`** — Unconstructible except via trusted functions; guarantees unique indices per thread.
2. **`DisjointSlice::get_mut()`** — Bounds-checked `Option<&mut T>`; `get_unchecked_mut()` is the explicit `unsafe` escape.
3. **`SharedArray` / `DynamicSharedArray`** — Intentionally `!Sync`; all access via `static mut` requires `unsafe`.
4. **Barriers, TMA, WGMMA, tcgen05** — All `unsafe` functions; caller ensures correct synchronization semantics.
5. **Atomics** — `unsafe impl Sync` on `DeviceAtomic*` types; ordering semantics match CUDA scoped atomics.
6. **`grid::sync()`** — Only valid in cooperative launches; non-cooperative launches deadlock.
7. **TMEM** — Must be explicitly deallocated before kernel exit; leaks are hard errors.

---

## Architecture Support Matrix

| Module | sm_50+ | sm_70+ | sm_80+ | sm_90+ | sm_100+ |
|--------|--------|--------|--------|--------|---------|
| `thread`, `warp` (basic) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `warp::match_*_sync` | — | ✅ | ✅ | ✅ | ✅ |
| `atomic` (scoped) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `grid::sync` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `barrier` (`mbarrier`) | — | — | ✅ | ✅ | ✅ |
| `tma` | — | — | — | ✅ | ✅ |
| `wgmma` | — | — | — | ✅ | ✅ |
| `cluster` | — | — | — | ✅ | ✅ |
| `tcgen05` | — | — | — | — | ✅ |

---

## Related Crates

| Crate | Role |
|-------|------|
| [`cuda-core`](../cuda-core/) | Host-side CUDA driver wrappers (contexts, streams, memory) |
| [`cuda-host`](../cuda-host/) | Higher-level host abstractions |
| [`cuda-macros`](../cuda-macros/) | Proc-macro implementations (`#[kernel]`, etc.) |
| [`cuda-async`](../cuda-async/) | `async`/`.await` integration with CUDA streams |

---

## License

Licensed under the [Apache-2.0](../../LICENSE-APACHE) license. Portions derived from NVIDIA CUDA samples are licensed under [LICENSE-NVIDIA](../../LICENSE-NVIDIA).
