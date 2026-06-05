# cuda-oxide Compilation Pipeline

This document traces the full path from Rust source code to executable PTX,
covering the five compiler-infrastructure crates: `mir-importer`, `mir-lower`,
`dialect-mir`, `dialect-nvvm`, and `llvm-export`.

---

## End-to-End Flow

```
Rust source code  (#[kernel] fn add(a: &[f32], b: &mut [f32]) { … })
       │
       ▼  rustc (Stable MIR extraction — rustc-codegen-cuda/src/collector.rs)
       │
╔══════════════════════════════════════════════════════════════════════════╗
║  mir-importer  (crates/mir-importer)                                     ║
║                                                                          ║
║  Step 1  translator::body::translate_body()                              ║
║          ├── emit_entry_allocas()   one mir.alloca per non-ZST local    ║
║          ├── For each reachable block:                                   ║
║          │     translator::block::translate_block()                     ║
║          │       ├── translator::statement::translate_statement()       ║
║          │       │     └── translator::rvalue::translate_rvalue()       ║
║          │       └── translator::terminator::translate_terminator()     ║
║          │             └── intrinsics/* (thread/warp/tma/wgmma/…)      ║
║          └── dialect-mir (alloca form): mir.alloca + mir.load/mir.store ║
║                                                                          ║
║  Step 2  pipeline::verify_operation()   Pliron verification pass        ║
║                                                                          ║
║  Step 3  pliron::opts::mem2reg()        alloca → SSA promotion          ║
║          └── dialect-mir (SSA form): no alloca, pure SSA values         ║
╚══════════════════════════════════════════════════════════════════════════╝
       │
       ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  mir-lower  (crates/mir-lower)                                           ║
║                                                                          ║
║  Step 4  lower_mir_to_llvm()  via pliron DialectConversion               ║
║          MirToLlvmConversionDriver dispatches per-op via op_cast        ║
║          ├── convert::types         MIR types → LLVM dialect types      ║
║          ├── convert::ops           arithmetic / memory / casts / …     ║
║          ├── convert::intrinsics    GPU intrinsics → NVVM calls / asm   ║
║          └── lowering::convert_func MirFuncOp → llvm.func               ║
║                 ├── flatten slice/struct args                            ║
║                 └── entry-block prologue (reconstruct aggregates)        ║
╚══════════════════════════════════════════════════════════════════════════╝
       │
       ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  llvm-export  (crates/llvm-export)                                       ║
║                                                                          ║
║  Step 5  export::export_module_with_externs()                            ║
║          ├── Header: datalayout + target triple                         ║
║          ├── Device extern declare statements (nvJitLink FFI)           ║
║          ├── export::function  — FuncOp → define/declare text           ║
║          │     ├── export::ops  — per-op textual emission               ║
║          │     ├── export::names — SSA value naming (stable across runs)║
║          │     └── block args → PHI nodes                               ║
║          ├── export::metadata — @llvm.used + !nvvm.annotations          ║
║          └── writes *.ll file                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
       │
       ▼
  llc (LLVM 21+, -march=nvptx64 -mcpu=sm_NNN)
       │
       ▼
  *.ptx  — textual PTX assembly, loaded by CUDA driver via cuModuleLoad
```

---

## Crates

### `dialect-mir` — Rust-semantic IR dialect

Defines the Pliron IR dialect that mirrors Rust MIR semantics.

| Module            | Contents                                         |
|-------------------|--------------------------------------------------|
| `ops/function`    | `mir.func` — function definition                |
| `ops/control_flow`| `mir.goto`, `mir.cond_branch`, `mir.return`, …  |
| `ops/memory`      | `mir.alloca`, `mir.load`, `mir.store`, `mir.ref`|
| `ops/arithmetic`  | `mir.add` … `mir.shr` (13 ops)                  |
| `ops/aggregate`   | `mir.insert_field`, `mir.extract_field`, …      |
| `ops/enum_ops`    | `mir.get_discriminant`, `mir.make_enum`, …      |
| `ops/cast`        | `mir.cast` with `cast_kind` attribute           |
| `ops/call`        | `mir.call`                                       |
| `types`           | `mir.ptr`, `mir.slice`, `mir.struct`, `mir.enum`|

**Key design choice — alloca + load/store model:**
Every non-ZST MIR local starts as a `mir.alloca` slot. Definitions emit
`mir.store` into the slot; uses emit `mir.load`. This sidesteps cross-block
SSA construction during translation. `mem2reg` converts back to SSA values
before LLVM lowering.

---

### `dialect-nvvm` — NVVM GPU intrinsic dialect

Models NVIDIA GPU intrinsics as Pliron IR operations. Each op lowers to one
or more LLVM intrinsic calls or inline PTX assembly fragments.

| Category   | Example ops                          | Arch       |
|------------|--------------------------------------|------------|
| `thread`   | `ReadPtxSregTidXOp`, `Barrier0Op`    | All        |
| `warp`     | `ShflSyncBflyI32Op`, `VoteSyncAnyOp` | All        |
| `atomic`   | `AtomicAddF32Op`, `AtomicCasI32Op`   | sm_70+     |
| `cluster`  | `ClusterSyncOp`, `MapaSharedCluster` | Hopper+    |
| `mbarrier` | `MbarrierInitOp`, `MbarrierTryWait`  | Hopper+    |
| `tma`      | `TmaG2sTile2dOp`, `TmaS2gTile2dOp`  | Hopper+    |
| `wgmma`    | `WgmmaFenceOp`, `WgmmaMmaAsyncF16`   | Hopper     |
| `tcgen05`  | `Tcgen05AllocOp`, `Tcgen05MmaOp`     | Blackwell+ |
| `stmatrix` | `StmatrixX4Op`                       | Hopper+    |

---

### `mir-importer` — Rust MIR → `dialect-mir`

Entry point: `translator::body::translate_body()` called per function from
`pipeline::run_pipeline()`.

**Intrinsic recognition:** GPU intrinsics arrive as MIR `TerminatorKind::Call`
to special symbols (e.g., `cuda_device::thread::thread_idx_x`). The
`translator/terminator/intrinsics/` subtree maps each symbol to the
corresponding `dialect-nvvm` op.

**Error handling:** `TranslationErr` (Unsupported / TypeError / InvalidOp)
wraps into pliron errors carrying MIR `Span` locations, so failures
show the Rust source location in the error message.

---

### `mir-lower` — `dialect-mir` → LLVM dialect

Entry point: `lower_mir_to_llvm()`, a pliron `DialectConversion` pass.

**Dispatch:** `MirToLlvmConversionDriver::rewrite()` special-cases four ops
that need pass-level state (`MirFuncOp`, `MirSharedAllocOp`,
`MirGlobalAllocOp`, `MirExternSharedOp`), then falls through to per-op
`MirToLlvmConversion::convert()` implementations.

**Function lowering:** Slice and struct function arguments are flattened into
scalar LLVM parameters. An entry-block prologue (`lowering::build_entry_prologue`)
reconstructs the aggregates from flattened values before branching to the
original MIR entry block.

**GPU intrinsic strategy:**
1. **LLVM intrinsic call** — simple ops with a direct NVVM intrinsic
   (e.g., `llvm.nvvm.read.ptx.sreg.tid.x` for thread X ID)
2. **Inline PTX assembly** — complex ops without direct intrinsics, or where
   PTX provides better control (e.g., `tcgen05.mma`, `wgmma.mma_async`)

---

### `llvm-export` — LLVM dialect → textual LLVM IR

Entry point: `export::export_module_with_externs()`.

**Backend configs:**
- `PtxExportConfig` — minimal datalayout for PTX via `llc`
- `NvvmExportConfig` — full NVPTX datalayout + `@llvm.used` +
  `!nvvm.annotations` + `!nvvmir.version` for libNVVM / nvJitLink

**PHI-node bridge:** pliron uses block arguments; LLVM IR uses PHI nodes.
The exporter builds a predecessor map during function emission and emits
`phi` instructions at the start of each block that has multiple predecessors.

**NVVM annotations** (`export/metadata.rs`): emits `!nvvm.annotations` for
kernels with cluster configuration (`cluster_dim_x/y/z`) or launch bounds
(`maxntidx`, `minctasm`).

---

## GPU Target Auto-Detection

After LLVM IR is written, `pipeline::generate_ptx()` scans the `.ll` file for
feature markers and selects an SM target:

| Feature marker in IR         | Target   | GPU family          |
|------------------------------|----------|---------------------|
| `tcgen05.*` / TMEM           | `sm_100a`| Blackwell datacenter|
| `g2s.tile … i1 1`(multicast) | `sm_100a`| Blackwell datacenter|
| `wgmma.fence.*`              | `sm_90a` | Hopper only         |
| `cp.async.bulk.tensor.*`     | `sm_100` | Hopper+ compatible  |
| `cluster_ctaid` / `cluster.sync`| `sm_90`| Hopper+ compatible  |
| (none of the above)          | `sm_80`  | Ampere+             |

Override with `CUDA_OXIDE_TARGET=<sm_NNN>`.

---

## NVVM IR / libdevice path

When the module calls `__nv_*` symbols (CUDA libdevice, e.g. `__nv_sinf`),
the pipeline automatically switches to NVVM IR mode:
- `llc` is skipped (PTX would have unresolved `__nv_*` externals)
- The `.ll` file is the final artifact
- The host build (see `examples/device_ffi_test/tools/`) compiles to LTOIR
  with libNVVM and links via nvJitLink + `libdevice.10.bc`

---

## Relevant Environment Variables

| Variable              | Effect                                         |
|-----------------------|------------------------------------------------|
| `CUDA_OXIDE_TARGET`   | Override auto-detected SM target               |
| `CUDA_OXIDE_LLC`      | Use a specific `llc` binary                    |
| `CUDA_OXIDE_VERBOSE`  | Extra progress/target-selection logging        |
| `CUDA_OXIDE_DUMP_MIR` | Dump `dialect-mir` IR after translation        |
