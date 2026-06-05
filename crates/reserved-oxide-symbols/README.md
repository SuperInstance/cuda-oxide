# reserved-oxide-symbols — INTERNAL

> **Not a public API.** This crate is `publish = false` and exists only to
> keep the macro side and the consumer side of the cuda-oxide naming
> contract in lockstep. The constants, builders, and predicates exposed
> here may change without notice between commits. External consumers
> should depend on `cuda-host`, `cuda-device`, or `cuda-macros` instead.

## Why Symbol Reservation Matters

cuda-oxide is a **custom codegen backend**: it replaces rustc's normal
LLVM-for-CPU path with a split pipeline that compiles some functions for
GPU PTX and others for host machine code. That split happens late — at
rustc's codegen phase — which means both host and device functions have
already been parsed, type-checked, and optimised together as a single MIR
body.

To decide which functions go to which backend, the codegen backend needs a
reliable way to recognise:

- Kernel entry points (`#[kernel]`)
- Device helper functions (`#[device]`)
- Device extern declarations (`#[device] extern "C"`)
- Closure monomorphisation helpers (generated for generic kernels)
- Constant statics (`#[constant]`)

The proc macros (`#[kernel]`, `#[device]`, `#[cuda_module]`) mark these by
renaming the item into a **reserved namespace**. This crate defines that
namespace and provides the single source of truth for every tool that needs
to read it.

Without a centralised contract, the macro side and the codegen side could
drift: a macro change that renames `cuda_oxide_kernel_foo` to
`cuda_oxide_kernel_bar` would break the backend's collector, leading to
silent omission of device code or mysterious "undefined kernel" errors at
runtime.

## The Naming Contract

Every internal symbol starts with `cuda_oxide_` and ends with a truncated
hash `246e25db_` (the first 8 hex chars of `sha256("cuda_oxide_ + rust")`).
The hash makes accidental collisions impossible — a user will never write
`fn cuda_oxide_kernel_246e25db_foo()` by accident.

| Constant | Value | Meaning |
|----------|-------|---------|
| `KERNEL_PREFIX` | `cuda_oxide_kernel_246e25db_` | `#[kernel]` entry point |
| `DEVICE_PREFIX` | `cuda_oxide_device_246e25db_` | `#[device]` helper function |
| `DEVICE_EXTERN_PREFIX` | `cuda_oxide_device_extern_246e25db_` | `#[device] extern "C"` declaration |
| `INSTANTIATE_PREFIX` | `cuda_oxide_instantiate_246e25db_` | Closure mono helper |
| `CONSTANT_PREFIX` | `cuda_oxide_const_246e25db_` | `#[constant]` static |

### Layered API

The crate exposes three concentric layers; pick the one that fits the call
site:

1. **Constants** (`KERNEL_PREFIX`, etc.) — raw prefix strings for code that
   needs the literal.
2. **Builders** (`kernel_symbol(base)`, etc.) — for the macro side that
   *produces* mangled names.
3. **Predicates and extractors** (`is_kernel_symbol`, `kernel_base_name`,
   etc.) — for the consumer side (collector, MIR lower, LLVM export) that
   *reads* mangled names and needs to recover the original base name.

### Host ↔ Device Alignment

The same symbol name must be understood identically on both sides of the
compilation wall:

- **Macro side** (`cuda-macros`): builds `cuda_oxide_kernel_246e25db_vecadd`
  when the user writes `#[kernel] fn vecadd`.
- **Codegen side** (`rustc-codegen-cuda`): scans MIR for that exact prefix,
  strips it to produce the PTX entry name `vecadd`, and emits it into the
  artifact bundle.
- **Host runtime** (`cuda-host`): reads the artifact bundle, finds the entry
  named `vecadd`, and passes it to `cuModuleGetFunction`.

If any of these three actors disagrees on the prefix or the stripping logic,
the kernel is invisible at runtime. Centralising the contract in this crate
eliminates that class of bug.

### Mutual-Exclusion Guarantee

`DEVICE_PREFIX` and `DEVICE_EXTERN_PREFIX` are **mutually exclusive**
substrings: a symbol containing one cannot contain the other. This is a
property of the hash suffix and is verified by a unit test. Consumers
therefore do **not** need the historical
`contains(DEVICE_PREFIX) && !contains(DEVICE_EXTERN_PREFIX)` ordering dance
— `is_device_symbol` handles the disambiguation internally.

## Why "reserved"

The `cuda_oxide_*` namespace is **reserved**: user code may not define
functions whose name starts with it. The `#[kernel]` and `#[device]` proc
macros enforce this at the source-code level via a compile-error guard, and
the hash suffix defends against the macro-bypass case (a plain function
literally named `cuda_oxide_kernel_foo`, no macro). Both defenses are
needed: the macro guard catches honest mistakes early at the source-code
level; the hash suffix makes the actually-mangled name collision-resistant
against any code path that bypasses the macro.
