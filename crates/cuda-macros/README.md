# cuda-macros

Procedural macros for writing CUDA kernels in Rust. Provides `#[kernel]` for GPU
entry points, `#[cuda_module]` for typed embedded-module loading, `#[device]` for
GPU helper functions, and lower-level `cuda_launch!` / `cuda_launch_async!`
macros.

## `#[kernel]` — GPU Kernel Entry Point

Marks a function as a CUDA kernel. The macro performs two transformations:

1. **Renames the function** into the reserved `cuda_oxide_kernel_<hash>_<name>`
   namespace with `#[no_mangle]` so the `rustc-codegen-cuda` backend can find it
   during MIR collection. The hash makes the prefix unguessable for user code;
   see `crates/reserved-oxide-symbols/` for the naming contract.

2. **Generates a marker struct** implementing `CudaKernel` (or `GenericCudaKernel`
   for generics) that carries the PTX entry-point name for host-side lookup.

### Non-generic kernel

```rust
use cuda_device::{kernel, DisjointSlice, thread};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

What the macro generates (conceptually):

```rust
#[unsafe(no_mangle)]
pub fn cuda_oxide_kernel_<hash>_vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    // body unchanged
}

pub struct __vecadd_CudaKernel;
impl cuda_host::CudaKernel for __vecadd_CudaKernel {
    const PTX_NAME: &'static str = "vecadd";
}
```

### Generic kernels

Generic kernels work in two modes.

**Mode 1 — call-site instantiation** (no explicit type list):

```rust
#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    let i = thread::index_1d();
    if let Some(o) = out.get_mut(i) {
        *o = input[i.get()] * factor;
    }
}
// Launch: module.scale::<f32>(&stream, config, factor, &input, &mut out)?
```

The macro generates:
- A `#[inline(always)]` wrapper with the original name for device-side calls.
- A `#[inline(never)]` prefixed entry point for the backend to collect.
- A `GenericCudaKernel` impl whose `ptx_name()` returns `scale_TID_<hex32>`,
  where `<hex32>` is the stable 128-bit type-id hash of the generic argument
tuple `(T,)`. Both backend and host compute the same hash from the same rustc
invocation, so the names match byte-for-byte.

**Mode 2 — explicit instantiation list**:

```rust
#[kernel(f32, i32)]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    // ...
}
```

Generates named entry points `scale_f32` and `scale_i32` with concrete wrapper
functions that call the generic body.

### Kernel attributes

| Attribute | Effect | PTX output |
|-----------|--------|------------|
| `#[launch_bounds(max_threads)]` | Max threads per block | `.maxntid N` |
| `#[launch_bounds(max_threads, min_blocks)]` | + min blocks per SM | `.maxntid N .minnctapersm M` |
| `#[cluster_launch(x, y, z)]` | Thread-block cluster dims | `.reqnctapercluster x, y, z` |

These must come **after** `#[kernel]`. They inject marker calls at the start of
the kernel body that the backend detects during MIR lowering.

### Reserved names

The macros refuse to compile any function whose name starts with `cuda_oxide_`.
That namespace is reserved for cuda-oxide-internal mangling. The check is enforced
at expansion time so the error points at the offending source line.

## `#[device]` — Device Helper Functions and Externs

Device functions run on the GPU but are not entry points. Works on both regular
functions and `extern "C"` blocks:

```rust
#[device]
pub fn magnitude(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

// Extern device functions (e.g. from libdevice or cuBLASDx)
#[device]
extern "C" {
    fn __nv_expf(x: f32) -> f32;
}
```

| Feature | `#[kernel]` | `#[device]` |
|---------|-------------|-------------|
| Entry point | Yes (PTX `.entry`) | No (PTX `.func`) |
| Can return values | No (must be `()`) | Yes |
| Callable from host | Via `#[cuda_module]` | No |
| Callable from device | Yes | Yes |

For non-generic device functions the macro adds `#[no_mangle]` and renames into
the reserved `cuda_oxide_device_<hash>_` prefix, then generates a thin
`#[inline(always)]` wrapper with the original name. For generic device functions
it uses `#[inline(never)]` on the prefixed function (so each monomorphization
appears as a distinct CGU item for the collector) and forwards via turbofish.

### Semantic markers

| Marker | LLVM attribute | Use case |
|--------|---------------|----------|
| `#[convergent]` | `convergent` | Sync primitives, warp/block collectives |
| `#[pure]` | `readnone` | Math with no side effects |
| `#[readonly]` | `readonly` | Read-only memory accessors |

These are pass-through attributes — no code transformation. The backend reads
them and applies the matching LLVM function attribute.

## `#[cuda_module]` — Typed Embedded Module Loading

Wrap an inline module containing `#[kernel]` functions to generate a typed loader
and per-kernel launch methods:

```rust
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
}
```

Generated API:

```rust
// Load the embedded artifact bundle for this crate.
let module = kernels::load(&ctx)?;

// Synchronous typed launch — rust-analyzer knows the name and argument types.
module.vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?;
```

When `cuda-host` is built with its `async` feature, the macro also emits:

```rust
let module = kernels::load_async(0)?;

// Borrowed async — ties lazy operation to buffer borrows.
module.vecadd_async(config, &a_dev, &b_dev, &mut c_dev)?.sync()?;

// Owned async — moves buffers into the operation, returns them on completion.
let (a_dev, b_dev, c_dev) = module
    .vecadd_async_owned(config, a_dev, b_dev, c_dev)?
    .await?;
```

Kernel parameters are mapped into host launch parameters:

| Kernel parameter | Host method parameter |
|------------------|-----------------------|
| `&[T]` | `&DeviceBuffer<T>` |
| `&mut [T]` | `&mut DeviceBuffer<T>` |
| `DisjointSlice<T>` | `&mut DeviceBuffer<T>` |
| `Copy` scalar, struct, closure, raw pointer | unchanged |

Because the launches are ordinary methods, rust-analyzer and rustc can complete
kernel names, show argument names, and type-check arguments before the program
runs. By-value arguments are copied into the CUDA launch packet through the
`KernelScalar` boundary; device slices are encoded as pointer-plus-length pairs.

`#[cuda_module]` also supports `#[constant]` statics inside the module. These
are rewritten with reserved `export_name` symbols and generate
`module.set_<name>(&stream, &value)` / `module.set_<name>_blocking(&value)`
host methods for updating constant memory.

## How closures get captured and passed to the GPU

A kernel can accept a closure as a generic parameter:

```rust
#[kernel]
pub fn map<F: Copy + Fn(u32) -> u32>(f: F, input: &[u32], mut out: DisjointSlice<u32>) {
    let i = thread::index_1d();
    if let Some(o) = out.get_mut(i) {
        *o = f(input[i.get()]);
    }
}
```

On the host, the closure is passed **by value** as a single aggregate argument.
The macro and backend cooperate so the entire closure struct is pushed once:

1. **Macro expansion**: `#[kernel]` detects the `Fn`/`FnMut`/`FnOnce` bound on a
generic parameter and generates an `instantiate_<name>` helper. This helper takes
`&F` (a reference to the closure) so the concrete anonymous type is bound to the
generic parameter at the call site, then forces rustc to emit a codegen-unit
entry for the monomorphized kernel via a volatile pointer write/read trick.

2. **PTX name**: The `GenericCudaKernel` impl hashes the tuple of generic
parameters (including the closure type) to produce `<base>_TID_<hex32>`. The
backend computes the identical hash, so host and device agree on the entry-point
name.

3. **Launch marshalling**: The closure value is pushed as a single byval scalar.
The backend emits a single `.param` declaration for the aggregate closure struct,
so one host push matches one device parameter. Zero-sized closures (no captures)
are omitted on both sides to keep parameter indices aligned.

```rust
// Host usage
module.map(move |x| x * 2, &input, &mut out)?;
```

Because the closure is `Copy`, each thread on the GPU receives its own copy of
the captured environment. The captures must be bitwise-copyable (no `String`,
`Vec`, or other heap types) and must make sense on the GPU (pointers to device
memory are fine; pointers to host memory are not).

## Complete example: defining and launching a kernel

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// 1. Define kernels in a #[cuda_module] inline module.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn saxpy(alpha: f32, x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d();
        if let Some(o) = out.get_mut(i) {
            *o = alpha * x[i.get()] + y[i.get()];
        }
    }

    #[kernel]
    pub fn map<F: Copy + Fn(f32) -> f32>(f: F, input: &[f32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d();
        if let Some(o) = out.get_mut(i) {
            *o = f(input[i.get()]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2. Set up CUDA context and stream.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let alpha = 2.0f32;
    let x_host = vec![1.0f32; N];
    let y_host = vec![3.0f32; N];

    // 3. Allocate and upload device buffers.
    let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
    let y_dev = DeviceBuffer::from_host(&stream, &y_host)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // 4. Load the embedded module.
    let module = kernels::load(&ctx)?;

    // 5. Launch a kernel with typed arguments.
    module.saxpy(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        alpha,
        &x_dev,
        &y_dev,
        &mut out_dev,
    )?;

    // 6. Launch a kernel with a closure.
    module.map(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        move |v| v.sqrt(),
        &x_dev,
        &mut out_dev,
    )?;

    // 7. Download results.
    let out_host: Vec<f32> = out_dev.to_host_vec(&stream)?;
    println!("First element: {}", out_host[0]);

    Ok(())
}
```

## `cuda_launch!` — Lower-Level Synchronous Launch

For code that predates `#[cuda_module]`, or when you need raw pointer control:

```rust
cuda_launch! {
    kernel: vecadd,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    cluster_dim: (4, 1, 1),       // optional, uses launch_kernel_ex
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
}
```

Argument forms:

| Syntax | Kernel parameter | Marshalling |
|--------|-----------------|-------------|
| `expr` | `T` (scalar) | `&mut value` as `*mut c_void` |
| `slice(buf)` | `&[T]` | Device pointer + length (two args) |
| `slice_mut(buf)` | `DisjointSlice<T>` | Device pointer + length (two args) |
| `move \|..\| body` | Closure `F` | Whole closure pushed by value |
| `\|..\| body` | Closure `F` | Whole closure struct (contains host references) |

## `cuda_launch_async!` — Lower-Level Async Launch

Returns an `AsyncKernelLaunch` implementing `DeviceOperation` for `cuda-async`
scheduling. Same argument forms as `cuda_launch!` but no `stream:` or
`cluster_dim:` fields.

```rust
let op = cuda_launch_async! {
    kernel: vecadd,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
};
op.sync()?;
```

## `gpu_printf!` — Device-Side Printf

Compiles to CUDA's `vprintf` with C vararg promotion rules. Format string must
use C-style specifiers.

```rust
gpu_printf!("thread %d: val = %f\n", tid as i32, val as f64);
```

## Source layout

```text
src/
├── lib.rs       # All proc-macro definitions (kernel, device, launch, etc.)
└── printf.rs    # gpu_printf! implementation
```

## Further reading

- [cuda-device](../cuda-device/) — re-exports these macros for convenience
- [cuda-host](../cuda-host/) — `CudaKernel` / `GenericCudaKernel` traits used by generated code
- [cuda-core](../cuda-core/) — `launch_kernel` / `launch_kernel_ex` called by generated code
