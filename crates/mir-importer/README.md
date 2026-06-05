# mir-importer

Rust MIR → `dialect-mir` translator and compilation pipeline for **cuda-oxide**.

This crate is cuda-oxide's compiler **frontend** (~24 KLOC). It consumes Rust's
Stable MIR (the Mid-level IR produced by `rustc`) and emits
[`dialect-mir`](../../dialect-mir/) — a pliron dialect that preserves Rust
semantics — then drives the rest of the GPU compilation pipeline down to PTX.

If you are reading this because you want to understand how Rust becomes GPU
assembly, start with [The Import Pipeline](#the-import-pipeline) and the
[Code Walkthrough](#code-walkthrough).

---

## Table of Contents

- [What is MIR?](#what-is-mir)
- [What is Pliron?](#what-is-pliron)
- [The Import Pipeline](#the-import-pipeline)
- [Type Mapping: Rust → IR](#type-mapping-rust--ir)
- [Translating Function Bodies](#translating-function-bodies)
- [Constants and Statics](#constants-and-statics)
- [Memory Model: Allocas, Load/Store, and Address Spaces](#memory-model)
- [Key Data Structures](#key-data-structures)
- [Architecture Diagram](#architecture-diagram)
- [Code Walkthrough: A Simple Function](#code-walkthrough-a-simple-function)
- [Module Structure](#module-structure)
- [Public API](#public-api)
- [Dependencies](#dependencies)

---

## What is MIR?

**MIR** (Mid-level Intermediate Representation) is the IR that `rustc` uses
after type-checking and before LLVM IR generation. If you have never worked
with compilers, think of it as a simplified, lower-level view of your Rust code
that still looks recognisably Rust-like.

A MIR **function body** consists of:

| Concept | Description | Example in Rust |
|---------|-------------|-----------------|
| **Locals** | Typed variables, numbered `_0`, `_1`, `_2`, … | `_1` might hold `x: i32` |
| **Basic blocks** | Sequences of statements ending in a *terminator* | `bb0`, `bb1`, … |
| **Statements** | Simple operations: assignments, storage markers | `_2 = _1` |
| **Terminators** | Control flow: `goto`, `call`, `return`, `switchInt` | `goto -> bb1` |
| **Rvalues** | Right-hand sides of assignments: `BinaryOp`, `Ref`, `Use`, … | `Add(_1, _2)` |
| **Places** | Lvalues: a local plus zero or more *projections* | `_3.1` (field 1 of tuple `_3`) |

MIR is **not** SSA (Static Single Assignment). A local like `_1` can be
assigned in `bb0` and read in `bb1`. The `mir-importer` bridges this gap by
emitting every non-ZST local as a stack slot (`mir.alloca`) and mediating all
defs/uses through `mir.store` / `mir.load`. A later `mem2reg` pass promotes
the slots back to SSA.

---

## What is Pliron?

**Pliron** is an MLIR-like compiler framework written in Rust. It provides:

- **Operations** — the equivalent of LLVM instructions, but structured (an op
can contain *regions*, which contain *blocks*, which contain more ops).
- **Dialects** — namespaces of related ops and types. `dialect-mir` models
Rust MIR; `dialect-nvvm` models CUDA GPU intrinsics; the LLVM dialect models
LLVM IR.
- **Types** — a type system with signless/signed/unsigned integers, floats,
structs, arrays, pointers with address spaces, and user-defined dialect types.
- **Passes** — transformations like `mem2reg` (promote allocas to SSA) and
`DialectConversion` (lower one dialect to another).

If MLIR is new to you: imagine LLVM IR, but every instruction can have
sub-regions (like a function body), and every instruction belongs to a dialect
so you can mix "Rust MIR ops" and "NVVM ops" in the same module during
lowering.

---

## The Import Pipeline

The end-to-end flow is:

```text
┌─────────────┐   ┌─────────────┐   ┌───────────┐   ┌──────────────┐   ┌─────────────┐
│ 1. Collect  │──▶│ 2. Translate│──▶│ 3. Verify │──▶│ 4. mem2reg   │──▶│ 5. Lower    │
│    MIR      │   │  MIR →      │   │  dialect- │   │  (alloca →   │   │  dialect-mir│
│  functions  │   │  dialect-mir│   │  mir      │   │   SSA)       │   │  → LLVM     │
└─────────────┘   └─────────────┘   └───────────┘   └──────────────┘   └─────────────┘
                                                                              │
┌─────────────┐   ┌─────────────┐   ┌───────────┐                            │
│ 8. Generate │◀──│ 7. Export   │◀──│ 6. Verify │◀───────────────────────────┘
│    PTX      │   │  LLVM IR    │   │  LLVM     │
│   (llc)     │   │   (.ll)     │   │  dialect  │
└─────────────┘   └─────────────┘   └───────────┘
```

1. **Collect** — `rustc-codegen-cuda` walks the crate and gathers monomorphized
   function instances (e.g. `add::<f32>`) tagged with `#[kernel]` or
   `#[device]`.
2. **Translate** — `mir-importer::translator::body::translate_body` turns each
   MIR body into a `mir.func` operation inside a `builtin.module`.
3. **Verify** — Pliron checks type consistency and structural invariants.
4. **mem2reg** — `pliron::opts::mem2reg` promotes scalar alloca slots to SSA
   values, eliminating most `mir.load`/`mir.store` traffic.
5. **Lower** — `mir-lower` converts `dialect-mir` → LLVM dialect via
   `DialectConversion`.
6. **Verify** — The LLVM dialect module is verified.
7. **Export** — `llvm-export` writes textual LLVM IR (`.ll`).
8. **Generate PTX** — `llc` compiles the LLVM IR to PTX assembly.

Steps 1 is handled by `rustc-codegen-cuda`; steps 2–8 are handled by
`mir_importer::pipeline::run_pipeline`.

---

## Type Mapping: Rust → IR

`translator::types::translate_type` converts Rust types to `dialect-mir` / pliron
types.

### Primitives

| Rust | `dialect-mir` / pliron |
|------|------------------------|
| `i8`–`i128`, `isize` | `IntegerType` (signed) |
| `u8`–`u128`, `usize` | `IntegerType` (unsigned) |
| `bool` | `i1` (signless) |
| `char` | `u32` |
| `f32` | `FP32Type` |
| `f64` | `FP64Type` |
| `f16` | `MirFP16Type` |
| `!` (never) | empty `MirTupleType` |

### Aggregates

| Rust | `dialect-mir` |
|------|---------------|
| `(A, B, C)` | `MirTupleType` |
| `[T; N]` | `MirArrayType` |
| `struct S { … }` | `MirStructType` (with layout from `ty.layout()`) |
| `enum E { … }` | `MirEnumType` (discriminant + variants) |
| closures | `MirStructType` (captures as fields) |

### Pointers and Slices

| Rust | `dialect-mir` |
|------|---------------|
| `*const T`, `*mut T`, `&T`, `&mut T` | `MirPtrType` (with address space) |
| `[T]`, `&[T]` | `MirSliceType` (fat pointer: data ptr + length) |

### CUDA-specific Types

| Rust type | Translation | Notes |
|-----------|-------------|-------|
| `SharedArray<T, N>` | ZST marker (`()`) | Actual memory is `MirSharedAllocOp` in `addrspace(3)` |
| `Barrier` | `u64` | `MirSharedAllocOp` in `addrspace(3)` |
| `DisjointSlice<T>` | `MirDisjointSliceType` | GPU fat pointer |
| `ThreadIndex` | `u64` | Type safety at Rust level only |
| `TmaDescriptor` | `[u64; 16]` | 128-byte opaque blob |
| `ConstantMemory<T>` | `MirPtrType` `addrspace(4)` | Constant (read-only) GPU memory |

Address spaces:

| Space | Number | Used for |
|-------|--------|----------|
| Generic | 0 | Default stack slots, generic pointers |
| Global | 1 | Ordinary `static` / `static mut` |
| Shared | 3 | `SharedArray`, `Barrier`, `__shared__` |
| Constant | 4 | `#[constant]` statics, read-only data |

---

## Translating Function Bodies

`body::translate_body` is the main entry point for function translation.

### Step 1 — Signature

MIR local `_0` is the return value; locals `_1` … `_N` are the function
arguments. The translator reads their types, converts them to pliron types, and
builds a `FunctionType`.

### Step 2 — Block creation

One pliron `BasicBlock` is created per MIR block. **Only the entry block**
carries arguments (the function parameters). Every other block is argument-less.
Cross-block data flow travels through the per-local alloca slots, not through
block arguments.

### Step 3 — Entry allocas

`emit_entry_allocas` walks every MIR local and emits one `mir.alloca` per
non-ZST local at the top of the entry block. Function arguments are immediately
`mir.store`'d into their slots so later blocks can load them.

### Step 4 — Block translation

Each reachable block is translated in index order by `block::translate_block`:

1. **Statements** → `statement::translate_statement`
2. **Terminator** → `terminator::translate_terminator`

Unwind-only cleanup blocks (unreachable on GPU) are patched with
`mir.unreachable` so pliron verification passes.

---

## Constants and Statics

`translator::rvalue::translate_operand` handles `Operand::Constant`.

### Scalar Constants

- **Integers** — parsed from the MIR constant allocation bytes, emitted as
  `mir.constant` with an `IntegerAttr`.
- **Floats** — `f32`/`f64` parsed from bytes or debug string; `f16` parsed as
  raw bits. Emitted as `mir.float_constant`.
- **Pointers** — `core::ptr::null()` becomes an integer `0` followed by a
  `mir.cast <PointerWithExposedProvenance>`. References to constant structs
  (e.g. `&(8..16)`) construct the struct value first, then use `mir.ref`.

### ZST Constants

Zero-sized constants (e.g. `PhantomData<T>`, unit `()`) are emitted as
`mir.construct_struct` / `mir.construct_tuple` with no operands.

### Static Variables

| Static kind | IR representation |
|-------------|-------------------|
| Ordinary `static` | `MirGlobalAllocOp` in `addrspace(1)` (global memory) |
| `#[constant] static` | `MirGlobalAllocOp` in `addrspace(4)` (constant memory) |
| `static mut` | Same as `static`, mutable pointer |
| `SharedArray` static | `MirSharedAllocOp` in `addrspace(3)` (shared memory) |
| `Barrier` static | `MirSharedAllocOp` in `addrspace(3)` |

All device-side statics are currently required to be zero-initialised;
host-populated initialisers are on the roadmap.

---

## Memory Model

### Alloca + Load/Store

MIR locals are not SSA — they can be mutated across blocks. The translator uses
an **alloca + load/store** model:

```text
Rust MIR (not SSA):                    dialect-mir (pre-mem2reg):

bb0: {                                 ^bb0(%arg0: i32):
    _1 = 42;                               %s1 = mir.alloca : !mir.ptr<i32>
    goto -> bb1;                           %c  = mir.constant 42 : i32
}                                          mir.store %c, %s1
                                           mir.goto ^bb1
bb1: {                                 ^bb1:
    _2 = _1;   // cross-block read!        %r = mir.load %s1 : i32
    return;                                mir.return %r : i32
}
```

Every non-ZST local gets a single `mir.alloca` in the entry block. Defs become
`mir.store`; uses become `mir.load`. `pliron::opts::mem2reg` later promotes
scalar slots back to SSA, so the above collapses to a direct `mir.return %c`.

### Address-Space Inference

Rust pointer types (`&mut T`, `*const T`) carry no address-space information.
On GPU, however, a local like `let p = &mut TILE_A[i]` (where `TILE_A` is a
`SharedArray`) produces an `addrspace(3)` pointer. If the slot for `p` were
typed with the generic address space, every store and load would need a
`cvta.shared` / `cvta.to.shared` round-trip.

`translator::values::SlotAddrSpaceMap` solves this by **pre-scanning the MIR
body** before alloca emission:

1. For each assignment into a local, classify the right-hand side.
2. If the RHS is confidently from shared memory (`SharedArray::index`,
   `DynamicSharedArray::get`, etc.), mark the local's slot as `Known(3)`.
3. If conflicting address spaces are observed, mark `Generic`.
4. `body::emit_entry_allocas` uses the inferred address space to override the
   Rust-declared type when creating the `mir.alloca`.

This is a monotone lattice fixed-point (`Uninit → Known(n) → Generic`) and
converges in at most `num_locals + 2` iterations.

---

## Key Data Structures

| Structure | Module | Role |
|-----------|--------|------|
| `ValueMap` | `values` | Maps each MIR local to its `mir.alloca` slot value. Provides `emit_alloca`, `load_local`, `store_local`. |
| `SlotAddrSpaceMap` | `values` | Inferred per-local address space for pointer slots. Pre-scans MIR before alloca emission. |
| `block_map: Vec<Ptr<BasicBlock>>` | `body` | Maps MIR block index → pliron block. Only entry block has arguments. |
| `CollectedFunction` | `pipeline` | A monomorphized function instance + kernel flag + export name. |
| `DeviceExternDecl` | `pipeline` | FFI-style external device symbol (for linking with LTOIR). |
| `PipelineConfig` | `pipeline` | Output directory, dump flags, NVVM IR mode, etc. |
| `TranslationErr` | `error` | Categorised errors (`Unsupported`, `TypeError`, `InvalidOp`) integrated with pliron's location system. |

---

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              mir-importer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         translator/                                  │   │
│  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ body   │──│ block  │──│ statement│──│ rvalue   │──│ terminator│  │   │
│  │  │        │  │        │  │          │  │          │  │          │  │   │
│  │  │•alloca │  │•stmts │  │•assign   │  │•binop    │  │•goto     │  │   │
│  │  │•blocks │  │•term   │  │•store    │  │•cast     │  │•call     │  │   │
│  │  │•kernel │  │        │  │•field-addr│ │•aggregate│  │•return   │  │   │
│  │  │ attrs  │  │        │  │          │  │•constant │  │•switch   │  │   │
│  │  └────────┘  └────────┘  └──────────┘  └──────────┘  └────┬─────┘  │   │
│  │                                                            │        │   │
│  │  ┌────────┐  ┌────────┐  ┌────────────────────────────────┐│        │   │
│  │  │ types  │  │ values │  │ terminator/intrinsics/         ││        │   │
│  │  │        │  │        │  │  • indexing (threadIdx, …)     ││        │   │
│  │  │•prims  │  │•ValueMap│ │  • sync (barriers, fences)     ││        │   │
│  │  │•aggreg │  │•SlotAS │  │  • warp (shuffle, vote)        ││        │   │
│  │  │•ptrs   │  │        │  │  • atomic (GPU atomics)        ││        │   │
│  │  │•cuda   │  │        │  │  • memory (shared, stmatrix)   ││        │   │
│  │  │ types  │  │        │  │  • wgmma / tcgen05 / tma / clc ││        │   │
│  │  └────────┘  └────────┘  └────────────────────────────────┘│        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         pipeline.rs                                  │   │
│  │  • run_pipeline — orchestrates translate → verify → mem2reg →       │   │
│  │    lower → verify → export → llc                                    │   │
│  │  • CollectedFunction, DeviceExternDecl, PipelineConfig               │   │
│  │  • GPU target auto-detection (sm_80 / sm_90 / sm_90a / sm_100a)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         error.rs                                     │   │
│  │  • TranslationErr — Unsupported / TypeError / InvalidOp             │   │
│  │  • TranslationResult — pliron::result::Result alias                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Walkthrough: A Simple Function

Let's follow a tiny Rust kernel through the importer.

```rust
#[kernel]
fn add_one(x: i32) -> i32 {
    x + 1
}
```

### 1. MIR (simplified)

```text
fn add_one(_1: i32) -> i32 {
    let mut _0: i32;          // return value
    let mut _2: i32;          // temporary

    bb0: {
        _2 = _1;              // copy argument
        _0 = Add(_2, const 1_i32);
        return;
    }
}
```

### 2. `body::translate_body`

- Extract signature: arg type `i32`, return type `i32`.
- Create `mir.func` operation with one region.
- Create one pliron block `^bb0` with one argument `%arg0: i32`.
- `emit_entry_allocas`:
  - `%s0 = mir.alloca : !mir.ptr<i32>`  (for `_0`, the return value)
  - `%s1 = mir.alloca : !mir.ptr<i32>`  (for `_1`, the argument)
  - `%s2 = mir.alloca : !mir.ptr<i32>`  (for `_2`, the temporary)
  - `mir.store %arg0, %s1`              (seed argument into its slot)

### 3. `block::translate_block` for `bb0`

**Statement 1:** `_2 = _1`
- `rvalue::translate_rvalue` sees `Use(Copy(_1))`.
- `translate_place` loads `_1`: `%v1 = mir.load %s1 : i32`.
- `statement::translate_statement` stores into `_2`: `mir.store %v1, %s2`.

**Statement 2:** `_0 = Add(_2, const 1_i32)`
- `translate_rvalue` translates the constant: `%c1 = mir.constant 1 : i32`.
- It loads `_2`: `%v2 = mir.load %s2 : i32`.
- It emits the add: `%sum = mir.add %v2, %c1 : i32`.
- The statement stores the result: `mir.store %sum, %s0`.

**Terminator:** `return`
- `translate_return` loads `_0`: `%ret = mir.load %s0 : i32`.
- Emits `mir.return %ret : i32`.

### 4. Resulting `dialect-mir` (pre-mem2reg)

```mlir
mir.func @add_one(%arg0: i32) -> i32 {
  %s0 = mir.alloca : !mir.ptr<i32>
  %s1 = mir.alloca : !mir.ptr<i32>
  %s2 = mir.alloca : !mir.ptr<i32>
  mir.store %arg0, %s1

  %v1 = mir.load %s1 : i32
  mir.store %v1, %s2

  %c1 = mir.constant 1 : i32
  %v2 = mir.load %s2 : i32
  %sum = mir.add %v2, %c1 : i32
  mir.store %sum, %s0

  %ret = mir.load %s0 : i32
  mir.return %ret : i32
}
```

### 5. After `mem2reg`

All allocas are promoted because no address escapes:

```mlir
mir.func @add_one(%arg0: i32) -> i32 {
  %sum = mir.add %arg0, %c1 : i32
  mir.return %sum : i32
}
```

### 6. Lowering to LLVM dialect → LLVM IR → PTX

`mir-lower` converts `mir.add` → `llvm.add`, `mir.return` → `llvm.return`, etc.
`llvm-export` writes `.ll`, and `llc -march=nvptx64 -mcpu=sm_80` produces the
final PTX.

---

## Module Structure

| File / Module | Purpose |
|---------------|---------|
| `lib.rs` | Crate root: re-exports, module declarations, top-level docs. |
| `pipeline.rs` | End-to-end compilation orchestration (`run_pipeline`). |
| `error.rs` | `TranslationErr` and `TranslationResult`. |
| `translator/mod.rs` | Dialect registration, `translate_function` entry point. |
| `translator/body.rs` | Function-level translation: signature, blocks, allocas, kernel attributes. |
| `translator/block.rs` | Basic block coordinator: statements then terminator. |
| `translator/statement.rs` | Assignment, storage markers, projections (`*ptr`, `.field`, `[i]`). |
| `translator/rvalue.rs` | Expressions: binary/unary ops, casts, aggregates, constants, places. |
| `translator/terminator/mod.rs` | Control flow: goto, call, return, switch, assert, drop. |
| `translator/terminator/helpers.rs` | Shared utilities: `emit_goto`, `emit_function_call`, `emit_nvvm_intrinsic`. |
| `translator/terminator/intrinsics/` | GPU intrinsic handlers (indexing, sync, warp, atomic, memory, wgmma, tcgen05, tma, clc, debug, bitops, float math, saturating). |
| `translator/types.rs` | Rust type → `dialect-mir` type conversion. |
| `translator/values.rs` | `ValueMap` and `SlotAddrSpaceMap` (slot management + addrspace inference). |
| `probe_pliron.rs` | Debug utilities (development only). |

---

## Public API

### Entry Point

```rust
use mir_importer::{run_pipeline, CollectedFunction, PipelineConfig};

let result = run_pipeline(&functions, &device_externs, &config)?;
// result.artifact_path  → .ptx or .ll
// result.ll_path        → LLVM IR
// result.ptx_path       → PTX assembly
// result.target         → e.g. "sm_90a"
```

### Key Types

| Type | Purpose |
|------|---------|
| `CollectedFunction` | MIR instance + kernel flag + export name |
| `DeviceExternDecl` | External device symbol declaration (for LTOIR linking) |
| `DeviceExternAttrs` | Convergent / pure / readonly markers |
| `PipelineConfig` | Output dir, verbosity, dump flags, NVVM IR mode |
| `CompilationResult` | Paths and target from a successful run |
| `PipelineError` | Categorised failure: `NoBody`, `Translation`, `Verification`, `Lowering`, `Export`, `PtxGeneration` |

---

## Dependencies

| Crate | Role |
|-------|------|
| [pliron](https://github.com/vaivaswatha/pliron) | MLIR-like IR framework |
| [dialect-mir](../../dialect-mir/) | Rust MIR dialect (ops, types, attributes) |
| [mir-lower](../../mir-lower/) | `dialect-mir` → LLVM dialect lowering |
| [llvm-export](../../llvm-export/) | LLVM dialect → textual `.ll` exporter |
| [dialect-nvvm](../../dialect-nvvm/) | NVVM GPU intrinsic ops |
| `rustc_public`, `rustc_public_bridge` | Stable MIR access (rustc internals) |

---

## Further Reading

- [rustc-codegen-cuda](../../rustc-codegen-cuda/) — the codegen backend that drives this crate.
- [dialect-mir](../../dialect-mir/) — the IR dialect this crate produces.
- [mir-lower](../../mir-lower/) — the lowering pass that consumes `dialect-mir`.
