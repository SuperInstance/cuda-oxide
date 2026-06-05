# cuda-oxide Architecture

This document is the authoritative reference for how the cuda-oxide workspace is structured, how its 18 crates compose, and how data flows from Rust source code to executing GPU kernels.

> **Scope:** This is the SuperInstance fork edition. It covers the full workspace as a systems-level compiler construction kit.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Workspace Composition](#workspace-composition)
3. [Compilation Pipeline](#compilation-pipeline)
4. [Runtime Architecture](#runtime-architecture)
5. [IR Dialect Design](#ir-dialect-design)
6. [Build System (cargo-oxide)](#build-system-cargo-oxide)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Cross-Cutting Concerns](#cross-cutting-concerns)

---

## Design Philosophy

 cuda-oxide treats GPU kernel compilation as a **first-class compiler pipeline** rather than a source-to-source transpiler or a DSL:

1. **Single-source, zero-overhead** — Host and device code live in the same `.rs` file. The compiler backend splits them automatically; no external build steps or annotation languages.
2. **Native MIR pipeline** — We ingest rustc's own Mid-level IR (Stable MIR), translate it into a custom Pliron dialect, lower through LLVM dialect, and emit NVPTX. This preserves Rust semantics (ownership, generics, monomorphization, closures) end-to-end.
3. **Crate-level modularity** — Each pipeline stage is a standalone crate with its own README, tests, and public API. You can reuse `dialect-mir` for other Rust-to-LLVM projects, or `cuda-core` as a safe CUDA driver wrapper without the compiler.
4. **Systems-grade runtime** — The host runtime is async-first, RAII-safe, and VMM-aware. It is designed for agent runtimes and long-running GPU services, not just one-shot benchmarks.

---

## Workspace Composition

The 18 crates are organized into four layers. Arrows indicate primary dependency direction (downward = depends on).

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: BUILD & ENTRYPOINT                         │
│  cargo-oxide  ──  cargo subcommand, orchestrates host + device builds       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│  LAYER 1: COMPILER BACKEND      │   │  LAYER 2: HOST RUNTIME              │
│  rustc-codegen-cuda             │   │  cuda-core, cuda-host, cuda-async   │
│  mir-importer, mir-lower        │   │  cuda-bindings, libnvvm-sys         │
│  dialect-mir, dialect-nvvm      │   │  nvjitlink-sys                      │
│  llvm-export                    │   │                                     │
└─────────────────────────────────┘   └─────────────────────────────────────┘
                    │                               │
                    │         ┌─────────────────────┘
                    │         │
                    │         ▼
                    │   ┌─────────────────────────────────────┐
                    │   │  LAYER 3: DEVICE RUNTIME            │
                    │   │  cuda-device, cuda-macros           │
                    │   └─────────────────────────────────────┘
                    │                   ▲
                    └───────────────────┘
           Compiler backend calls device intrinsics;
           device macros generate metadata consumed by backend.
```

### Dependency Graph (simplified)

```text
rustc-codegen-cuda
    ├── mir-importer
    │   ├── dialect-mir
    │   ├── dialect-nvvm
    │   ├── mir-lower
    │   │   ├── dialect-mir
    │   │   ├── dialect-nvvm
    │   │   └── llvm-export
    │   │       └── pliron-llvm
    │   └── llvm-export
    └── cuda-host
        ├── cuda-core
        │   └── cuda-bindings
        ├── cuda-async
        │   ├── cuda-core
        │   └── cuda-bindings
        ├── libnvvm-sys
        └── nvjitlink-sys

cuda-device
    └── cuda-macros

reserved-oxide-symbols  (internal, used by cuda-macros + rustc-codegen-cuda)
oxide-artifacts         (used by cuda-core, cuda-host)
```

---

## Compilation Pipeline

The compiler path is the heart of cuda-oxide. It runs inside `rustc-codegen-cuda` when rustc processes a crate containing `#[kernel]` functions.

### Phase-by-Phase Flow

```text
Phase 0:  rustc Stable MIR extraction
          └─>  rustc_codegen_cuda::collector gathers #[kernel] fns
               and their transitive device dependencies.

Phase 1:  mir-importer::translator::body::translate_body()
          └─>  One mir.alloca per non-ZST MIR local
               Loop over reachable BasicBlocks:
                 translate_statement() → rvalue translation
                 translate_terminator() → intrinsic recognition
          └─>  dialect-mir (alloca form)

Phase 2:  mir-importer::pipeline::verify_operation()
          └─>  Pliron structural verification (SSA dominance, type consistency)

Phase 3:  pliron::opts::mem2reg()
          └─>  Promote alloca slots to SSA values
          └─>  dialect-mir (SSA form)

Phase 4:  mir-lower::lower_mir_to_llvm()
          └─>  Pliron DialectConversion pass
               • MirFuncOp → llvm.func  (flatten args, entry prologue)
               • Mir arithmetic → llvm.add, llvm.mul, …
               • Mir memory → llvm.load, llvm.store, llvm.alloca
               • GPU intrinsics → dialect-nvvm ops → llvm.nvvm.* calls
          └─>  LLVM dialect module

Phase 5:  llvm-export::export_module_with_externs()
          └─>  Textual LLVM IR (.ll)
               • datalayout + nvptx64-unknown-unknown target triple
               • PHI nodes for block-argument predecessors
               • !nvvm.annotations metadata (kernels, launch bounds)

Phase 6:  llc (system LLVM tool, LLVM 21+)
          └─>  PTX assembly (.ptx)
               • Target: sm_80 (Ampere+) | sm_90 (Hopper) | sm_90a (Hopper wgmma)
                 | sm_100a (Blackwell tcgen05)
               • Auto-detected from IR feature markers

Phase 7:  oxide-artifacts embedding
          └─>  PTX (or LTOIR) is embedded as a COFF/ELF section in the host binary
               via oxide-artifacts container format.

Phase 8:  Host runtime load
          └─>  cuda-host::load_module() reads embedded artifact at runtime
               → cuModuleLoadDataEx → typed launcher invocation
```

### Alternative Path: NVVM IR / libdevice

When kernels call `__nv_*` math functions (CUDA libdevice), the pipeline switches:

```text
Phase 5 (LLVM IR .ll) ──>  libnvvm-sys  (libNVVM front-end)
                                    │
                                    ▼
                              LTOIR / PTX  (with libdevice.10.bc linked)
                                    │
                                    ▼
                              nvjitlink-sys  (nvJitLink linker)
                                    │
                                    ▼
                              Final cubin / PTX  loaded by CUDA driver
```

This path skips `llc` because libdevice externals must be resolved by NVIDIA's own compiler stack.

---

## Runtime Architecture

The runtime is split into three tiers: **explicit synchronous**, **typed launching**, and **composable async**.

### Synchronous Host Runtime (cuda-core)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  cuda-core  ──  Safe RAII wrappers around cu* driver API                     │
│                                                                              │
│  CudaContext  ──  wraps CUcontext  (primary ctx management)                  │
│  CudaStream   ──  wraps CUstream   (default + user-created async streams)    │
│  DeviceBuffer<T>  ──  device memory with D2H / H2D / D2D transfers           │
│  PinnedBuffer<T>  ──  page-locked host memory for async transfers            │
│  CudaModule   ──  cuModuleLoad from embedded or file artifact                │
│  CudaEvent    ──  timing and cross-stream synchronization                    │
│                                                                              │
│  VMM APIs (Virtual Memory Management)                                        │
│  ├── PhysicalAllocation  ──  cuMemCreate                                     │
│  ├── VirtualReservation ──  cuMemAddressReserve                              │
│  └── Mapping            ──  cuMemMap / cuMemSetAccess                        │
│                                                                              │
│  Peer access: can_access_peer, enable_peer_access                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Typed Launch Layer (cuda-host)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  cuda-host  ──  Generated by #[cuda_module] macro + manual launch helpers    │
│                                                                              │
│  #[cuda_module] mod kernels {                                                │
│      #[kernel] fn add(a: &[f32], b: &mut [f32]) { ... }                     │
│  }                                                                           │
│                                                                              │
│  Generates at compile time:                                                  │
│  • kernels::load(&ctx) -> Result<Module, ...>                               │
│  • module.add(&stream, config, a, b) -> Result<(), ...>                     │
│    └── scalarizes slices into ptr+len, packs into CUDA kernel params         │
│                                                                              │
│  LTOIR support:                                                              │
│  • cuda_host::ltoir::compile_nvvm_ir()  → libNVVM → LTOIR                   │
│  • cuda_host::ltoir::link_modules()     → nvJitLink → cubin/PTX             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Composable Async Layer (cuda-async)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  cuda-async  ──  Lazy GPU operations + futures                               │
│                                                                              │
│  DeviceOperation<R>  ──  opaque handle to a deferred GPU computation         │
│    ├── .sync()  →  blocks host until completion, returns R                  │
│    └── .await   →  async Rust future, yields R when GPU signals done        │
│                                                                              │
│  DeviceBox<T>  ──  GPU-resident value with host-side future access           │
│  DeviceFuture<T>  ──  promise-like handle for cross-stream dependencies      │
│                                                                              │
│  Stream graph (implicit):                                                    │
│  • Operations on the same stream execute in submission order                 │
│  • Operations on different streams run concurrently                          │
│  • Explicit CudaEvent barriers for cross-stream ordering                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Device Runtime (cuda-device)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  cuda-device  ──  #![no_std] GPU intrinsics (runs on the SM, not the host)   │
│                                                                              │
│  Thread indexing                                                             │
│  ├── thread::index_1d() / index_2d() / index_3d()                           │
│  └── ThreadIdx, BlockIdx, BlockDim, GridDim                                 │
│                                                                              │
│  Memory abstractions                                                         │
│  ├── DisjointSlice<T>  ──  non-overlapping mutable slice guarantee          │
│  ├── SharedMemory<T>   ──  `__shared__` with scoped lifetime                │
│  └── DeviceGlobal<T>   ──  `__device__` persistent storage                  │
│                                                                              │
│  Synchronization                                                             │
│  ├── barrier::syncthreads()                                                  │
│  ├── cluster::sync()      (Hopper+ thread block clusters)                   │
│  └── warp::*              shuffle, vote, reduce                             │
│                                                                              │
│  Tensor & async copy (Hopper+)                                               │
│  ├── tma::copy_2d_tile_g2s() / s2g()                                        │
│  ├── wgmma::*             (WGMMA fence, mma_async, Hopper)                  │
│  └── tcgen05::*           (TMEM alloc, MMA, Blackwell)                      │
│                                                                              │
│  Atomics (6 types × 3 scopes × 5 orderings)                                  │
│  └── atomic::add_relaxed(), atomic::cas_acquire(), …                        │
│                                                                              │
│  Debug                                                                       │
│  └── gpu_printf!("idx=%d\n", thread::index_1d().get());                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## IR Dialect Design

The compiler uses [Pliron](https://github.com/pliron-org/pliron), an MLIR-like IR framework written in Rust. cuda-oxide defines two custom dialects.

### `dialect-mir` — Rust-semantic IR

Mirrors rustc MIR semantics at the statement/rvalue level. Every non-ZST local starts as `mir.alloca`; `mem2reg` promotes to SSA before LLVM lowering.

| Op category | Key ops | Purpose |
|-------------|---------|---------|
| **Function** | `mir.func` | Function definition with MIR-local slot table |
| **Control flow** | `mir.goto`, `mir.cond_branch`, `mir.return`, `mir.unreachable` | Basic-block terminators |
| **Memory** | `mir.alloca`, `mir.load`, `mir.store`, `mir.ref` | Stack slots, derefs, borrows |
| **Arithmetic** | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, `mir.shl`, `mir.shr`, … | Integer and float ops |
| **Aggregate** | `mir.insert_field`, `mir.extract_field`, `mir.make_tuple` | Struct/tuple construction |
| **Enum** | `mir.get_discriminant`, `mir.make_enum`, `mir.set_discriminant` | Rust enum operations |
| **Cast** | `mir.cast` (with `cast_kind` attr) | `as` casts, pointer casts, transmutes |
| **Call** | `mir.call` | Direct and indirect function calls |

**Type system:**
- `mir.ptr<T>` — raw pointer
- `mir.slice<T>` — fat pointer (ptr + len)
- `mir.struct<...>` — aggregate type
- `mir.enum<...>` — discriminated union

### `dialect-nvvm` — GPU intrinsic IR

Models NVIDIA-specific hardware operations. Each op lowers to either an `@llvm.nvvm.*` intrinsic call or inline PTX assembly.

| Category | Example ops | Lowering target | Architecture |
|----------|-------------|-----------------|--------------|
| `thread` | `ReadPtxSregTidXOp`, `ReadPtxSregCtaidYOp` | `@llvm.nvvm.read.ptx.sreg.*` | All |
| `warp` | `ShflSyncBflyI32Op`, `VoteSyncAnyOp` | `@llvm.nvvm.shfl.sync.bfly.*` | All |
| `atomic` | `AtomicAddF32Op`, `AtomicCasI32Op` | Inline PTX `atom.*` | sm_70+ |
| `cluster` | `ClusterSyncOp`, `MapaSharedCluster` | Inline PTX `cluster.sync` | Hopper+ |
| `mbarrier` | `MbarrierInitOp`, `MbarrierTryWait` | Inline PTX `mbarrier.*` | Hopper+ |
| `tma` | `TmaG2sTile2dOp`, `TmaS2gTile2dOp` | Inline PTX `cp.async.bulk.tensor.*` | Hopper+ |
| `wgmma` | `WgmmaFenceOp`, `WgmmaMmaAsyncF16` | Inline PTX `wgmma.*` | Hopper |
| `tcgen05` | `Tcgen05AllocOp`, `Tcgen05MmaOp` | Inline PTX `tcgen05.*` | Blackwell+ |
| `stmatrix` | `StmatrixX4Op` | Inline PTX `stmatrix.*` | Hopper+ |

**Design note:** We use inline PTX for complex ops because LLVM's NVPTX backend does not expose intrinsics for every Hopper/Blackwell instruction. The inline assembly fragments are verified against PTX ISA specs and guarded by SM-target checks in `mir-lower`.

---

## Build System (cargo-oxide)

`cargo-oxide` replaces the traditional `xtask` pattern with a first-class Cargo subcommand that understands the dual host/device nature of cuda-oxide projects.

### Responsibilities

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  cargo-oxide  ──  Build orchestration                                        │
│                                                                              │
│  1. Environment discovery                                                   │
│     • Locate CUDA toolkit (CUDA_PATH, /usr/local/cuda)                      │
│     • Locate LLVM (rustup llvm-tools, llc-22, llc-21 on PATH)               │
│     • Verify rust-src + rustc-dev components                                │
│                                                                              │
│  2. Backend provisioning                                                    │
│     • Fetch/build rustc_codegen_cuda dylib on first run                     │
│     • Cache per-toolchain to avoid rebuilds                                 │
│                                                                              │
│  3. Dual compilation                                                        │
│     • Host: standard cargo build with codegen backend registered            │
│     • Device: backend extracts MIR, runs pipeline, emits PTX/LTOIR          │
│     • Embed device artifact into host binary via oxide-artifacts            │
│                                                                              │
│  4. Developer utilities                                                     │
│     • cargo oxide run <example>    ──  build + run                          │
│     • cargo oxide pipeline <ex>    ──  print MIR → PTX pipeline stages      │
│     • cargo oxide debug <ex>       ──  launch under cuda-gdb                │
│     • cargo oxide doctor           ──  prerequisite check                   │
│     • cargo oxide new <name>       ──  bootstrap a new cuda-oxide project   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Build Flow

```text
User runs: cargo oxide run vecadd

         ┌─────────────────┐
         │  cargo-oxide    │
         └────────┬────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ doctor  │ │ backend │ │ cargo   │
│ check   │ │ build   │ │ build   │
└─────────┘ └────┬────┘ └────┬────┘
                 │           │
                 ▼           ▼
        ┌─────────────────────────┐
        │  rustc_codegen_cuda     │
        │  • host code → LLVM     │
        │  • device code → PTX    │
        └─────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────┐
        │  oxide-artifacts embed  │
        │  PTX into host binary   │
        └─────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────┐
        │  ./target/debug/vecadd  │
        │  (host exe + embedded   │
        │   device code)          │
        └─────────────────────────┘
```

---

## Data Flow Diagrams

### Example: `vecadd` Kernel (End-to-End)

```rust
#[kernel]
fn vecadd(a: &[f32], mut b: DisjointSlice<f32>) {
    let i = thread::index_1d().get();
    if let Some(out) = b.get_mut(i) {
        *out = a[i] + 1.0;
    }
}
```

```text
Rust source
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ rustc_codegen_cuda::collector                                               │
│   Identifies vecadd as #[kernel]                                            │
│   Extracts Stable MIR Body for vecadd                                       │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ mir-importer::translator::body::translate_body(vecadd_body)                 │
│                                                                              │
│   MIR locals:                                                                │
│   _0  → mir.alloca !mir.ptr<f32>   (return slot, elided)                   │
│   _1  → mir.alloca !mir.slice<f32> (arg a)                                  │
│   _2  → mir.alloca !mir.slice<f32> (arg b)                                  │
│   _3  → mir.alloca !mir.u64        (i)                                      │
│   _4  → mir.alloca !mir.ptr<f32>   (out)                                    │
│                                                                              │
│   BB0:                                                                       │
│     _3 = cuda_device::thread::index_1d()   → intrinsic → dialect-nvvm      │
│     _3 = _3.get()                          → mir.call                       │
│     _4 = _2.get_mut(_3)                    → mir.call + DisjointSlice logic │
│     mir.cond_branch(_4 is Some?, BB1, BB2)                                  │
│   BB1:                                                                       │
│     _5 = *_4                               → mir.load                       │
│     _6 = *_1.offset(_3)                    → mir.load + mir.add (ptr offset)│
│     _7 = _6 + 1.0f32                       → mir.add                        │
│     mir.store _5 ← _7                                                       │
│     mir.goto BB2                                                             │
│   BB2:                                                                       │
│     mir.return                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ pliron::opts::mem2reg()                                                     │
│   Promotes _1, _2, _3, _4, _5, _6, _7 from alloca slots to SSA values       │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ mir-lower::lower_mir_to_llvm()                                              │
│                                                                              │
│   mir.func vecadd → llvm.func @vecadd(...)                                  │
│   • flatten slice args: (ptr, len) pairs                                    │
│   • thread::index_1d() → @llvm.nvvm.read.ptx.sreg.tid.x + blockDim calc     │
│   • mir.add f32 → llvm.fadd                                                 │
│   • mir.load / mir.store → llvm.load / llvm.store                           │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ llvm-export::export_module_with_externs()                                   │
│   target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8..."                       │
│   target triple = "nvptx64-unknown-unknown"                                 │
│                                                                              │
│   define ptx_kernel void @vecadd(float* %a_ptr, i64 %a_len,                 │
│                                  float* %b_ptr, i64 %b_len) {               │
│     ...                                                                      │
│   }                                                                          │
│                                                                              │
│   !nvvm.annotations = !{!0}                                                  │
│   !0 = !{void (float*, i64, float*, i64)* @vecadd, !"kernel", i32 1}       │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  llc -march=nvptx64 -mcpu=sm_80
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ .version 8.4                                                                │
│ .target sm_80                                                               │
│ .entry vecadd(                                                             │
│   .param .u64 vecadd_param_0,                                              │
│   .param .u64 vecadd_param_1,                                              │
│   .param .u64 vecadd_param_2,                                              │
│   .param .u64 vecadd_param_3)                                              │
│ {                                                                            │
│   ... PTX instructions ...                                                 │
│ }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ oxide-artifacts embeds PTX into host binary COFF/ELF section               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Host program runs: kernels::load(&ctx)?                                     │
│   → cuda_host reads embedded section                                        │
│   → cuModuleLoadDataEx(ptx_bytes)                                           │
│   → module.vecadd(&stream, config, &a, &mut b)?                             │
│     → cuLaunchKernel(@vecadd, grid, block, params...)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Cross-Cutting Concerns

### Error Handling Strategy

- **Compiler crates** (`mir-importer`, `mir-lower`, `llvm-export`) use `thiserror` and pliron's diagnostic system. Errors carry MIR `Span` locations so they point back to the original Rust source line.
- **Runtime crates** (`cuda-core`, `cuda-host`, `cuda-async`) use `anyhow` at the API surface and `thiserror` for typed error enums internally. Every `cu*` call result is checked.
- **Macro crates** (`cuda-macros`) use `proc_macro::Diagnostic` (nightly) or `compile_error!` for syntax errors.

### Memory Safety Model

| Layer | Guarantee | Mechanism |
|-------|-----------|-----------|
| `cuda-bindings` | None — raw FFI | `unsafe` functions, bindgen output |
| `cuda-core` | RAII + drop guards | `DeviceBuffer<T>` owns allocation; `CudaContext` owns primary ctx |
| `cuda-host` | Type-safe launches | Generated methods validate slice lengths at launch time |
| `cuda-device` | `#![no_std]` + no allocator | Static analysis; `DisjointSlice` proves non-overlap at type level |
| `cuda-async` | Structured concurrency | `DeviceOperation` drop waits for GPU completion to prevent UAF |

### Testing Pyramid

```text
        ┌─────────────┐
        │   E2E       │  60+ examples in rustc-codegen-cuda/examples/
        │  examples   │  cargo oxide run <example> — full host+device+runtime
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  Integration │  cuda-core tests, cuda-device trybuild tests
        │    tests     │  mir-lower round-trip tests
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │    Unit      │  Per-crate unit tests (cargo test -p <crate>)
        │    tests     │  dialect op verification, export snapshot tests
        └──────┬──────┘
               │
        ┌──────┴──────┐
        │  Fuzz / diff │  rustlantis → random MIR → compare rustc vs cuda-oxide
        │   codegen    │  crates/fuzzer/
        └─────────────┘
```

### Environment Variables

| Variable | Effect |
|----------|--------|
| `CUDA_OXIDE_TARGET` | Override auto-detected SM target (`sm_80`, `sm_90`, `sm_90a`, `sm_100a`) |
| `CUDA_OXIDE_LLC` | Pin a specific `llc` binary |
| `CUDA_OXIDE_VERBOSE` | Extra logging during pipeline execution |
| `CUDA_OXIDE_DUMP_MIR` | Dump `dialect-mir` IR after translation to stderr |
| `CUDA_PATH` | CUDA toolkit root (fallback: `/usr/local/cuda`) |

---

## Glossary

| Term | Meaning |
|------|---------|
| **MIR** | Rust Mid-level IR — rustc's analysis/optimization IR before LLVM |
| **Pliron** | MLIR-like IR framework in Rust (pliron-org/pliron) |
| **Dialect** | A namespace of ops and types in Pliron (e.g., `dialect-mir`, `dialect-nvvm`) |
| **NVVM** | NVIDIA LLVM IR dialect — LLVM IR with NVPTX target and NVIDIA intrinsics |
| **PTX** | Parallel Thread Execution — NVIDIA's ISA-level assembly language |
| **LTOIR** | Link-Time Optimization IR — NVIDIA's bitcode format for device linking |
| **SM** | Streaming Multiprocessor — NVIDIA GPU compute unit (sm_80 = Ampere, sm_90 = Hopper, sm_100a = Blackwell) |
| **TMA** | Tensor Memory Accelerator — async copy engine on Hopper+ |
| **WGMMA** | Warp-Group Matrix Multiply Accumulate — Hopper tensor core API |
| **TCGEN05** | Tensor Core GENeration 05 — Blackwell tensor memory + MMA |
| **HMM** | Heterogeneous Memory Management — CUDA unified memory |

---

*Last updated: 2026-06-05 — SuperInstance fork edition.*
