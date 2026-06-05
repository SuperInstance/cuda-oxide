# dialect-mir

MIR dialect for [Pliron](https://github.com/vaivaswatha/pliron) — 6.7K LOC, 144 pub items.

This crate defines the first compiler-specific IR in the cuda-oxide pipeline. It models Rust's Mid-level Intermediate Representation (MIR) as a Pliron *dialect*, preserving Rust-semantic types and operations before they are lowered to LLVM.

---

## What is a "Dialect"?

In MLIR (and Pliron, its Rust equivalent), a **dialect** is a self-contained namespace of operations, types, and attributes that model a specific abstraction level. The `dialect-mir` crate registers:

- **Operations** (`mir.add`, `mir.load`, `mir.func`, …) — what the program *does*
- **Types** (`mir.ptr`, `mir.slice`, `mir.struct`, …) — what values *are*
- **Attributes** (`mir.cast_kind`, `mir.field_index`, …) — metadata attached to ops

Dialects compose: `mir-lower` consumes `dialect-mir` ops and emits `pliron-llvm` ops. `dialect-nvvm` (GPU intrinsics) co-exists in the same IR and is lowered alongside `dialect-mir`.

---

## Pipeline Position

```text
Rust source code  (#[kernel] fn add(a: &[f32], b: &mut [f32]) { … })
       │
       ▼  rustc (Stable MIR extraction)
       │
┌──────────────┐
│ mir-importer │  (rustc MIR → dialect-mir alloca form)
└──────┬───────┘
       │
       ▼  pliron::opts::mem2reg()  (alloca → SSA promotion)
       │
┌──────────────┐
│  dialect-mir │  ◄── THIS CRATE (SSA form, no alloca)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  mir-lower   │  (dialect-mir → LLVM dialect)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ llvm-export  │  (LLVM dialect → textual .ll)
└──────┬───────┘
       │
       ▼
     *.ptx
```

---

## Types

The dialect defines seven Rust-specific types. All extend pliron's builtin integer/float types.

| Type | Pliron Syntax | Description |
|------|---------------|-------------|
| `MirPtrType` | `mir.ptr<T, mutable: bool, addrspace: u32>` | Typed pointer with mutability and NVPTX address space |
| `MirSliceType` | `mir.slice<T>` | Fat pointer `{ ptr, i64 }` for `&[T]` |
| `MirDisjointSliceType` | `mir.disjoint_slice<T>` | Same layout as slice, thread-local access semantics |
| `MirStructType` | `mir.struct<"Name", [fields…], [types…], layout…>` | Named struct with rustc field offsets for ABI correctness |
| `MirEnumType` | `mir.enum<"Name", discr_ty, [variants…]>` | Discriminant + payload union |
| `MirArrayType` | `mir.array<T, N>` | Fixed-size array `[T; N]` |
| `MirTupleType` | `mir.tuple<T1, T2, …>` | Heterogeneous tuple |
| `MirFP16Type` | `mir.fp16` | IEEE-754 half precision |

### Address Spaces

Pointers and slices carry an NVPTX address space that determines which GPU memory region they access:

| Space | ID | PTX Qualifier | Use |
|-------|----|---------------|-----|
| Generic | 0 | (none) | Default, resolved at runtime |
| Global | 1 | `.global` | Device VRAM |
| Shared | 3 | `.shared` | Per-block scratchpad |
| Constant | 4 | `.const` | Read-only cached |
| Local | 5 | `.local` | Per-thread stack / spill |
| TensorMem | 6 | `.param` | Blackwell+ tcgen05 operands |

`MirStructType` stores the exact layout from rustc (`field_offsets`, `mem_to_decl`, `total_size`) so that `#[repr(Rust)]` structs match the host ABI even when fields are reordered.

---

## Operations

54 operations across 11 modules. Every op implements pliron's `Verify` trait so that importer bugs are caught immediately rather than deferred to LLVM.

### Arithmetic (`ops/arithmetic.rs`)

Integer and floating-point math, bitwise logic, and shifts.

| Op | Description |
|----|-------------|
| `mir.add` | Integer or float addition |
| `mir.sub` | Subtraction |
| `mir.mul` | Multiplication |
| `mir.div` | Division (signed / unsigned inferred at lowering) |
| `mir.rem` | Remainder |
| `mir.neg` | Unary negation |
| `mir.not` | Bitwise NOT |
| `mir.and`, `mir.or`, `mir.xor` | Bitwise logic |
| `mir.shl`, `mir.shr` | Left / right shift |
| `mir.checked_add` | Addition with overflow flag tuple |

### Comparison (`ops/comparison.rs`)

Relational and equality comparisons. All yield `i1`.

| Op | Description |
|----|-------------|
| `mir.lt`, `mir.le`, `mir.gt`, `mir.ge` | Ordered comparison |
| `mir.eq`, `mir.ne` | Equality |

### Memory (`ops/memory.rs`)

Load, store, allocate, and address arithmetic.

| Op | Description |
|----|-------------|
| `mir.alloca` | Stack slot allocation (promotable to SSA by `mem2reg`) |
| `mir.load` | Load from pointer |
| `mir.store` | Store to pointer |
| `mir.ref` | Take address of a local (`&x`) |
| `mir.assign` | Direct value assignment |
| `mir.ptr_offset` | Pointer arithmetic (`ptr.add(offset)`) |
| `mir.shared_alloc` | Static shared memory allocation |
| `mir.global_alloc` | Global device memory allocation |
| `mir.extern_shared` | Dynamic shared memory (size from launch config) |

`MirAllocaOp` implements `PromotableAllocationInterface`; `MirLoadOp` / `MirStoreOp` implement `PromotableOpInterface`. This lets pliron's `mem2reg` pass promote scalar stack slots back to SSA values, erasing the alloca form entirely before lowering.

### Control Flow (`ops/control_flow.rs`)

| Op | Description |
|----|-------------|
| `mir.return` | Function return |
| `mir.goto` | Unconditional branch |
| `mir.cond_branch` | Conditional branch (`i1` condition) |
| `mir.assert` | Runtime assertion with message |
| `mir.unreachable` | Unreachable code marker |

### Function (`ops/function.rs`)

| Op | Description |
|----|-------------|
| `mir.func` | Function definition (single region, basic blocks inside) |

### Aggregate (`ops/aggregate.rs`)

Struct, tuple, and array manipulation.

| Op | Description |
|----|-------------|
| `mir.construct_aggregate` | Build a struct / tuple / array from fields |
| `mir.extract_field` | Extract field by constant index |
| `mir.insert_field` | Insert value into field by index |
| `mir.field_addr` | Address of a struct field (`&s.field`) |
| `mir.extract_array_element` | Array element by runtime index |

### Enum (`ops/enum_ops.rs`)

| Op | Description |
|----|-------------|
| `mir.construct_enum` | Build a specific variant with payload |
| `mir.get_discriminant` | Read discriminant integer |
| `mir.enum_payload` | Extract payload fields from a variant |

### Other Modules

| Module | Ops | Description |
|--------|-----|-------------|
| `cast` | `mir.cast` | Type conversions (kind tracked via `MirCastKindAttr`: `IntToFloat`, `PtrToPtr`, `Transmute`, …) |
| `constants` | `mir.constant`, `mir.float_constant`, `mir.undef` | Integer, float, and undef literals |
| `storage` | `mir.storage_live`, `mir.storage_dead` | Lifetime markers (erased during lowering) |
| `call` | `mir.call` | Direct and indirect function calls |

---

## How Ops Compose to Represent GPU Programs

A simple kernel like

```rust
#[kernel]
fn saxpy(a: f32, x: &[f32], y: &mut [f32]) {
    let i = thread::index_1d().get();
    y[i] = a * x[i] + y[i];
}
```

translates into `dialect-mir` as a `mir.func` containing basic blocks with the following ops:

1. **Thread-index intrinsic** — `mir-importer` recognizes `thread::index_1d()` and emits a `dialect-nvvm` op (lowered later to `llvm.nvvm.read.ptx.sreg.tid.x`).
2. **Bounds check** — `mir.lt` + `mir.cond_branch` to guard the store.
3. **Slice element access** — `mir.extract_field` on the slice struct to get the `ptr` and `len`, then `mir.ptr_offset` and `mir.load`.
4. **Arithmetic** — `mir.mul` (`a * x[i]`), `mir.add` (`+ y[i]`).
5. **Store result** — `mir.store` through a `mir.ptr_offset` into `y`.
6. **Return** — `mir.return`.

The slice argument `y: &mut [f32]` is represented as `mir.slice<f32>` (a struct of `{ ptr, i64 }`). The mutable reference itself becomes a `mir.ptr<mir.slice<f32>, mutable, addrspace: 0>` when passed by reference from the host.

---

## Attributes

Four domain-specific attributes avoid overloaded `IntegerAttr`:

| Attribute | Rust Type | Purpose |
|-----------|-----------|---------|
| `mir.cast_kind` | `MirCastKindAttr` | Preserves Rust cast intent for lowering |
| `mir.mutability` | `MutabilityAttr` | `&` vs `&mut` |
| `mir.field_index` | `FieldIndexAttr` | Structural field index |
| `mir.variant_index` | `VariantIndexAttr` | Enum variant index |

---

## Registration

```rust
use pliron::context::Context;
use dialect_mir::register;

let mut ctx = Context::new();
register(&mut ctx);  // Registers all ops, types, and attributes
```

---

## Further Reading

- [mir-importer](../mir-importer/) — translates rustc MIR → `dialect-mir`
- [mir-lower](../mir-lower/) — lowers `dialect-mir` → LLVM dialect
- [dialect-nvvm](../dialect-nvvm/) — GPU intrinsics that coexist in the same IR
- [llvm-export](../llvm-export/) — textual LLVM IR exporter
