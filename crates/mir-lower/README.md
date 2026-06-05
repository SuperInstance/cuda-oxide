# mir-lower

`dialect-mir` → LLVM dialect lowering pass — 14K LOC.

This crate is the bridge between Rust semantics and LLVM's target-agnostic IR. It converts every `dialect-mir` operation (and co-located `dialect-nvvm` GPU intrinsics) into equivalent operations from the `pliron-llvm` LLVM dialect. After this pass, the module contains only LLVM-dialect ops and can be exported to textual LLVM IR by [`llvm-export`](../llvm-export/).

---

## What is "Lowering"?

In compiler terminology, **lowering** is the translation of a higher-level intermediate representation into a lower-level one, preserving semantics while exposing more machine-level detail.

In cuda-oxide:

| Level | Representation | Concepts |
|-------|---------------|----------|
| High | `dialect-mir` | Rust types (`enum`, `slice`, `&mut T`), Rust operations (`checked_add`, `field_addr`) |
| Low | LLVM dialect | Signless integers, raw pointers, PHI nodes, NVVM intrinsics, inline PTX |

`mir-lower` performs this translation operation-by-operation using pliron's `DialectConversion` framework, which handles IR walking, def-before-use ordering, type conversion, and block-argument patching automatically.

---

## Pipeline Position

```text
Rust source code
       │
       ▼
┌──────────────┐
│   rustc      │  (extracts Stable MIR)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ mir-importer │  (Stable MIR → dialect-mir alloca form)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   mem2reg    │  (alloca → SSA promotion)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  mir-lower   │  ◄── THIS CRATE (dialect-mir → LLVM dialect)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ llvm-export  │  (LLVM dialect → textual .ll)
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

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    MirToLlvmConversionDriver                  │
│              (pliron DialectConversion implementation)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│  convert   │  │  convert   │  │  convert   │
│   types    │  │    ops     │  │ intrinsics │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │
      ▼               ▼               ▼
 LLVM dialect    LLVM dialect    LLVM dialect
   types           ops           ops + asm
```

### Entry Point

```rust
use mir_lower::lower_mir_to_llvm;

lower_mir_to_llvm(&mut ctx, module_op)?;
```

`MirToLlvmConversionDriver` implements pliron's `DialectConversion` trait. Its `rewrite` method dispatches each op via the `MirToLlvmConversion` op interface. Four ops need pass-level CUDA state and are special-cased:

- `MirFuncOp` → `convert_func` (kernel attrs, entry prologue, region inlining)
- `MirSharedAllocOp` → shared-memory global deduplication
- `MirGlobalAllocOp` → device-global deduplication
- `MirExternSharedOp` → dynamic shared memory + alignment tracking

All other ops dispatch generically through `op_cast::<dyn MirToLlvmConversion>`.

---

## Type Conversion

`convert::types` translates `dialect-mir` types to LLVM dialect types.

| `dialect-mir` Type | LLVM dialect Type | Notes |
|--------------------|-------------------|-------|
| `IntegerType` (signed/unsigned) | `IntegerType` (signless) | Width preserved; signedness moves to ops |
| `MirFP16Type` | `half` | Rust `f16` → LLVM `half` |
| `FP32Type`, `FP64Type` | `float`, `double` | Pass-through |
| `MirPtrType` | `PointerType` | Address space preserved |
| `MirSliceType` | `StructType { ptr, i64 }` | Fat pointer (pointer + length) |
| `MirDisjointSliceType` | `StructType { ptr, i64 }` | Same layout as slice |
| `MirTupleType` | `StructType` | ZST fields dropped |
| `MirStructType` | `StructType` | Explicit padding arrays inserted when rustc layout is known |
| `MirEnumType` | `StructType { discr, fields… }` | Discriminant + max payload |
| `MirArrayType` | `ArrayType` | Element type converted |

### Function Type Conversion

Function types undergo ABI flattening to match the C/GPU kernel calling convention:

- **Slice args** → flattened to `(ptr, i64)` scalar parameters
- **Struct args** → flattened to individual scalar fields
- **Empty tuple return** → `void`

An entry-block prologue reconstructs the aggregates from flattened values via `llvm.insertvalue` before branching to the original MIR entry block.

---

## Operation Lowering

### Arithmetic → LLVM

| MIR Op | LLVM Op | Notes |
|--------|---------|-------|
| `mir.add` | `llvm.add` / `llvm.fadd` | Integer vs float auto-detected |
| `mir.sub` | `llvm.sub` / `llvm.fsub` | |
| `mir.mul` | `llvm.mul` / `llvm.fmul` | |
| `mir.div` | `llvm.sdiv` / `llvm.udiv` / `llvm.fdiv` | Signedness read from pre-conversion type |
| `mir.rem` | `llvm.srem` / `llvm.urem` / `llvm.frem` | |
| `mir.neg` | `llvm.sub 0, x` / `llvm.fneg` | |
| `mir.not` | `llvm.xor x, -1` | |
| `mir.and`, `or`, `xor` | `llvm.and`, `or`, `xor` | |
| `mir.shl` | `llvm.shl` | Shift amount masked |
| `mir.shr` | `llvm.lshr` / `llvm.ashr` | Logical vs arithmetic from signedness |
| `mir.checked_add` | `llvm.add` + `llvm.extractvalue` overflow tuple | |

### Comparison → LLVM

| MIR Op | LLVM Op | Predicate Selection |
|--------|---------|---------------------|
| `mir.eq`, `ne` | `llvm.icmp` / `llvm.fcmp` | `eq`/`ne` for int; `oeq`/`une` for float |
| `mir.lt`, `le`, `gt`, `ge` | `llvm.icmp` / `llvm.fcmp` | Signed (`slt`) vs unsigned (`ult`) from pre-conversion type |

### Memory → LLVM

| MIR Op | LLVM Op | Notes |
|--------|---------|-------|
| `mir.alloca` | `llvm.alloca` | Promoted away by mem2reg before reaching here |
| `mir.load` | `llvm.load` | Alignment inferred from type |
| `mir.store` | `llvm.store` | |
| `mir.ref` | `llvm.alloca` + `llvm.store` | Address-of local |
| `mir.assign` | `llvm.store` | Direct copy |
| `mir.ptr_offset` | `llvm.getelementptr` | |

### Control Flow → LLVM

| MIR Op | LLVM Op |
|--------|---------|
| `mir.return` | `llvm.return` |
| `mir.goto` | `llvm.br` |
| `mir.cond_branch` | `llvm.cond_br` |
| `mir.assert` | `llvm.cond_br` → trap block |
| `mir.unreachable` | `llvm.unreachable` |
| `mir.storage_live` / `dead` | Erased (no-op) |

### Aggregate → LLVM

| MIR Op | LLVM Op |
|--------|---------|
| `mir.construct_aggregate` | `llvm.insertvalue` chain |
| `mir.extract_field` | `llvm.extractvalue` |
| `mir.insert_field` | `llvm.insertvalue` |
| `mir.field_addr` | `llvm.getelementptr` |
| `mir.extract_array_element` | `llvm.extractvalue` (constant index) or GEP+load |

### Enum → LLVM

| MIR Op | LLVM Op |
|--------|---------|
| `mir.construct_enum` | `llvm.insertvalue` (discriminant + payload) |
| `mir.get_discriminant` | `llvm.extractvalue` [0] |
| `mir.enum_payload` | `llvm.extractvalue` (field indices) |

### Cast → LLVM

`mir.cast` carries a `MirCastKindAttr` that selects the LLVM instruction:

| Cast Kind | LLVM Instruction |
|-----------|------------------|
| `IntToInt` | `llvm.sext` / `llvm.zext` / `llvm.trunc` |
| `IntToFloat` | `llvm.sitofp` / `llvm.uitofp` |
| `FloatToInt` | `llvm.fptosi` / `llvm.fptoui` |
| `FloatToFloat` | `llvm.fpext` / `llvm.fptrunc` |
| `PtrToPtr` | `llvm.bitcast` / `llvm.addrspacecast` |
| `Transmute` | `llvm.bitcast` |

---

## GPU-Specific Lowering

GPU intrinsics from `dialect-nvvm` are lowered in `convert::intrinsics`. Two strategies are used:

1. **LLVM NVVM intrinsic calls** — for well-supported ops (thread IDs, barriers, atomics, TMA).
2. **Inline PTX assembly** — for ops without LLVM intrinsics, or where exact PTX control is needed (WGMMA, tcgen05, stmatrix). Uses the `convergent` attribute to prevent LLVM from moving warp-synchronous ops across control flow.

### Thread and Block Indices

| NVVM Op | Lowered To |
|---------|-----------|
| `nvvm.read_ptx_sreg_tid_x` | `call i32 @llvm.nvvm.read.ptx.sreg.tid.x()` |
| `nvvm.read_ptx_sreg_ctaid_y` | `call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()` |
| `nvvm.read_ptx_sreg_ntid_z` | `call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()` |

### Barriers and Fences

| NVVM Op | Lowered To |
|---------|-----------|
| `nvvm.barrier0` | `call void @llvm.nvvm.barrier0()` |
| `threadfence_block` | Inline PTX `membar.cta` |
| `threadfence` | Inline PTX `membar.gl` |
| `threadfence_system` | Inline PTX `membar.sys` |

### Warp Operations

| NVVM Op | Lowered To |
|---------|-----------|
| `shfl_sync_bfly` | `call i32 @llvm.nvvm.shfl.sync.bfly.i32(...)` |
| `vote_sync_any` | `call i32 @llvm.nvvm.vote.sync.any(...)` |
| `lane_id` | `call i32 @llvm.nvvm.read.ptx.sreg.laneid()` |

### Atomics

Scoped GPU atomics and `core::sync::atomic` operations lower to LLVM `atomicrmw` / `cmpxchg` with NVPTX address spaces and syncscopes:

| Operation | LLVM Equivalent | Min Arch |
|-----------|-----------------|----------|
| `atomic_add_f32` | `atomicrmw fadd` addrspace(1) | sm_70+ |
| `atomic_cas_i32` | `cmpxchg` | sm_70+ |
| Scoped atomics | `atomicrmw` + `syncscope("device")` / `syncscope("block")` | sm_70+ |

### Shared Memory

- **Static** (`SharedArray<T, N>`): Lowered to `@__shared_*` globals in address space 3. `SharedGlobalsMap` deduplicates identical allocations across the module.
- **Dynamic** (`DynamicSharedArray<T>`): Lowered to `@__dynamic_smem_*` extern globals. `DynamicSmemAlignmentMap` tracks the maximum alignment per kernel so the runtime can size the allocation correctly.

### Advanced GPU Intrinsics

| Category | Examples | Strategy | Min SM |
|----------|----------|----------|--------|
| `mbarrier` | `mbarrier.init`, `mbarrier.try_wait` | LLVM intrinsics | sm_90+ |
| `cluster` | `cluster.sync`, DSMEM ring exchange | Inline PTX | sm_90+ |
| `tma` | `tma.g2s.tile.2d`, `tma.s2g.tile.2d` | Inline PTX | sm_90+ |
| `wgmma` | `wgmma.mma_async`, `wgmma.fence` | Inline PTX | sm_90 |
| `tcgen05` | `tcgen05.mma`, `tcgen05.alloc` | Inline PTX | sm_100+ |
| `stmatrix` | `stmatrix.x4` | Inline PTX | sm_90+ |
| `clc` | Cluster Launch Control | LLVM intrinsics | sm_100+ |

---

## Intrinsic Handling: Math Functions and libdevice

When Rust code calls math functions (`sin`, `exp`, `sqrt`, …), the importer emits `mir.call` to `__nv_*` symbols (CUDA libdevice, e.g. `__nv_sinf`). During lowering:

- The call is preserved as an LLVM `call` to the `__nv_*` symbol.
- The pipeline detects these symbols and switches to **NVVM IR mode**:
  - `llc` is skipped (PTX would have unresolved externals).
  - The `.ll` file is passed to libNVVM + nvJitLink, which links against `libdevice.10.bc`.

This path is also used for **Device FFI**: Rust kernels calling C++ CCCL functions via LTOIR. `llvm-export` emits `declare` statements for extern symbols; nvJitLink resolves them at link time.

---

## Source Layout

```text
src/
├── lib.rs                          # DialectConversion driver + lower_mir_to_llvm()
├── lowering.rs                     # convert_func: MirFuncOp → llvm.func
├── conversion_interface.rs         # MirToLlvmConversion op interface
├── type_conversion_interface.rs    # MirConvertibleType trait
├── context.rs                      # SharedGlobalsMap, DynamicSmemAlignmentMap
├── helpers.rs                      # Constants, intrinsic declarations, utilities
├── convert/
│   ├── types.rs                    # MIR → LLVM type conversion
│   ├── interface_impls.rs          # Op interface impls dispatching to converters
│   ├── type_interface_impls.rs     # Type interface impls
│   └── ops/
│       ├── arithmetic.rs           # Math, bitwise, shifts
│       ├── memory.rs               # Load, store, alloca, GEP
│       ├── control_flow.rs         # Branches, returns, asserts
│       ├── constants.rs            # Integer/float/undef literals
│       ├── cast.rs                 # Type conversions
│       ├── aggregate.rs            # Struct, tuple, array ops
│       └── call.rs                 # Function calls with arg flattening
│   └── intrinsics/
│       ├── basic.rs                # Thread/block IDs, barrier
│       ├── warp.rs                 # Shuffle, vote
│       ├── atomic.rs               # GPU atomics
│       ├── mbarrier.rs             # Async barriers
│       ├── cluster.rs              # Block clusters
│       ├── tma.rs                  # Tensor Memory Accelerator
│       ├── wgmma.rs                # Warpgroup MMA
│       ├── tcgen05.rs              # 5th-gen Tensor Cores
│       ├── stmatrix.rs             # Shared-memory matrix store
│       ├── clc.rs                  # Cluster Launch Control
│       ├── debug.rs                # Clock, trap, printf
│       └── common.rs               # Shared helpers
```

---

## Further Reading

- [dialect-mir](../dialect-mir/) — source dialect (Rust MIR semantics)
- [dialect-nvvm](../dialect-nvvm/) — GPU intrinsic ops consumed by this crate
- [llvm-export](../llvm-export/) — exports LLVM dialect to textual `.ll`
- [PIPELINE.md](../../PIPELINE.md) — end-to-end compilation flow
