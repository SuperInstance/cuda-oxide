# llvm-export

Pliron LLVM dialect → textual LLVM IR exporter — 3.3K LOC.

This crate sits at the back-end of the cuda-oxide compiler pipeline. It receives an LLVM-dialect module (produced by [`mir-lower`](../mir-lower/)) and serialises it to textual LLVM IR (`.ll`) that `llc` can assemble into PTX. It also re-exports [`pliron-llvm`](https://github.com/vaivaswatha/pliron) and adds a few GPU-specific extensions that upstream lacks.

---

## Pipeline Position

```text
Rust source code
       │
       ▼
┌──────────────┐
│   rustc      │  (Stable MIR extraction)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ mir-importer │  (Stable MIR → dialect-mir)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  mir-lower   │  (dialect-mir → LLVM dialect)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ llvm-export  │  ◄── THIS CRATE (LLVM dialect → textual .ll)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     llc      │  (LLVM IR → PTX)
└──────────────┘
       │
       ▼
     *.ptx
```

---

## What Lives Here

### 1. Pliron-LLVM Re-Exports

The LLVM dialect *modeling* (ops like `llvm.add`, types like `llvm.ptr`, attributes, and op-interfaces) lives upstream in `pliron-llvm`. This crate re-exports it so that `llvm_export::{ops, types, attributes, op_interfaces}` paths keep resolving after the upstream migration.

### 2. GPU-Specific Extensions

| Extension | Why it lives here |
|-----------|-------------------|
| Named address spaces | `pliron-llvm` stores a raw `u32`; we define `generic=0`, `global=1`, `shared=3`, `constant=4`, `local=5`, `tmem=6` |
| `PointerTypeExt` | Convenience constructors `get_shared()`, `get_global()`, `is_tmem()` |
| `LlvmSyncScope` | Enum for `System` / `Device` / `Block` atomic scopes (upstream uses `Option<String>`) |
| `InlineAsmOpExt` | `new_convergent(...)` helper used across `mir-lower` for warp-sync PTX |
| `GlobalOpExt` | Explicit alignment on `GlobalOp` (stored in generic attr dict) |
| fp16 bit helpers | `fp16_attr_from_bits()` / `fp16_attr_to_bits()` for `apfloat::Half` |

### 3. Textual LLVM IR Exporter

The `export` module walks an LLVM-dialect module and prints stable, deterministic `.ll` text without depending on `llvm-sys`.

---

## How Pliron IR Maps to LLVM IR

### Module Structure

A pliron `ModuleOp` becomes an LLVM IR module with a standard header:

```llvm
; ModuleID = 'module'
source_filename = "kernels"
target datalayout = "e-i64:64-i128:128..."
target triple = "nvptx64-nvidia-cuda"
```

The exporter then emits, in order:

1. **Device extern declarations** — `declare` statements for symbols resolved by nvJitLink (Device FFI / libdevice).
2. **Global variables** — Shared memory (`@__shared_*`) and dynamic shared memory (`@__dynamic_smem_*`) with address-space qualifiers.
3. **Function definitions** — `define` for kernels and helpers; `declare` for intrinsics.
4. **Attribute groups** — Shared LLVM attributes referenced by functions.
5. **Metadata** — `!nvvm.annotations`, `!nvvmir.version`, `@llvm.used`.

### Function Definitions and Basic Blocks

Pliron represents functions as `llvm.func` ops containing a region of `BasicBlock`s. Each basic block holds a sequence of operations.

**Block arguments vs PHI nodes:**

Pliron uses **block arguments** (a functional SSA form). LLVM IR uses **PHI nodes** at the start of a block. The exporter bridges this gap:

1. During function emission, it builds a `PredecessorMap` — for each block, which blocks branch to it.
2. When emitting a block with multiple predecessors, it inserts `phi` instructions at the top, collecting the incoming values from each predecessor.

Example:

```text
Pliron:
  ^bb1(%x: i32):
    llvm.add %x, %c1

  ^bb2(%y: i32):
    llvm.br ^bb3(%y)

  ^bb3(%z: i32):
    ...
```

```llvm
LLVM IR:
  br label %bb3

bb2:
  br label %bb3

bb3:
  %z = phi i32 [ %x, %bb1 ], [ %y, %bb2 ]
  ...
```

### SSA Value Naming

The exporter runs a pre-pass (`export::names`) that assigns deterministic names to all anonymous SSA values. This ensures that the textual IR is **bit-for-bit stable across runs**, which is essential for reproducible builds and caching.

---

## Type Translation

The exporter prints LLVM dialect types to their LLVM IR textual equivalents:

| Pliron LLVM Dialect Type | LLVM IR Text |
|--------------------------|--------------|
| `IntegerType` | `iN` (e.g., `i32`, `i64`) |
| `PointerType` (addrspace 0) | `ptr` |
| `PointerType` (addrspace N) | `ptr addrspace(N)` |
| `VoidType` | `void` |
| `HalfType` | `half` |
| `FP32Type` | `float` |
| `FP64Type` | `double` |
| `StructType` | `{ T1, T2, … }` |
| `ArrayType` | `[N x T]` |
| `VectorType` | `<N x T>` |

Pointers in non-generic address spaces carry the `addrspace(N)` qualifier, which the NVPTX backend uses to emit `.global`, `.shared`, `.local`, etc.

---

## Metadata and Attributes

### Kernel Annotations (`!nvvm.annotations`)

Every `#[kernel]` function receives metadata that tells the CUDA driver the symbol is a kernel entry point:

```llvm
!0 = !{ ptr @saxpy, !"kernel", i32 1 }
!nvvm.annotations = !{ !0 }
```

### Launch Bounds

Kernels annotated with `#[launch_bounds(max_threads)]` emit `maxntidx` metadata:

```llvm
!1 = !{ ptr @saxpy, !"kernel", i32 1, !"maxntidx", i32 256 }
```

With `min_blocks`:

```llvm
!1 = !{ ptr @saxpy, !"kernel", i32 1, !"maxntidx", i32 256, !"minctasm", i32 2 }
```

### Cluster Configuration

Kernels using thread-block clusters emit `cluster_dim_x/y/z`:

```llvm
!2 = !{ ptr @cluster_kernel, !"kernel", i32 1,
        !"cluster_dim_x", i32 2,
        !"cluster_dim_y", i32 1,
        !"cluster_dim_z", i32 1 }
```

### `@llvm.used`

In NVVM IR mode, `@llvm.used` preserves kernel symbols so they survive LTO linking:

```llvm
@llvm.used = appending global [1 x ptr] [ ptr @saxpy ], section "llvm.metadata"
```

### `!nvvmir.version`

NVVM IR mode also emits the NVVM IR version metadata node required by libNVVM:

```llvm
!nvvmir.version = !{ !{ i32 2, i32 0 } }
```

---

## Backend Configurations

The exporter supports two configurations via the `ExportBackendConfig` trait:

| Configuration | Use Case | `@llvm.used` | `!nvvmir.version` | `!nvvm.annotations` |
|---------------|----------|--------------|-------------------|---------------------|
| `PtxExportConfig` | Standard PTX via `llc` | No | No | Launch bounds only |
| `NvvmExportConfig` | NVVM IR for libNVVM / LTOIR | Yes | Yes | All kernels |

```rust
use llvm_export::export::{export_module_to_string, export_module_with_externs, NvvmExportConfig};

// Standard PTX path
let ll = export_module_to_string(&ctx, &module)?;

// NVVM IR path (for libNVVM + nvJitLink + libdevice)
let nvvm_ir = export_module_with_externs(&ctx, &module, &device_externs, &NvvmExportConfig)?;
```

### Target Configuration

| Setting | Value |
|---------|-------|
| Target triple | `nvptx64-nvidia-cuda` |
| Data layout | 64-bit pointers, 128-bit `i128` alignment |
| PTX version | 8.7+ (for `sm_120`) |

---

## Source Layout

```text
src/
├── lib.rs              # Re-exports pliron-llvm + GPU extensions
└── export/
    ├── mod.rs          # Export entry points (export_module_to_string, export_module_with_externs)
    ├── config.rs       # PtxExportConfig / NvvmExportConfig / ExportBackendConfig trait
    ├── module.rs       # Module-level emission flow
    ├── function.rs     # Function bodies, block-arg → PHI translation
    ├── ops.rs          # Per-operation textual emission
    ├── types.rs        # Type printing
    ├── literals.rs     # Constant / literal formatting
    ├── metadata.rs     # !nvvm.annotations, launch bounds, cluster config, @llvm.used
    ├── externs.rs      # Device extern declaration types
    ├── names.rs        # Deterministic SSA value / symbol naming
    └── state.rs        # Export state tracking (predecessor maps, kernel bookkeeping)
```

---

## Further Reading

- [dialect-mir](../dialect-mir/) — source dialect (lowering input)
- [dialect-nvvm](../dialect-nvvm/) — GPU intrinsics
- [mir-lower](../mir-lower/) — produces the LLVM dialect this crate consumes
- [PIPELINE.md](../../PIPELINE.md) — end-to-end compilation flow
