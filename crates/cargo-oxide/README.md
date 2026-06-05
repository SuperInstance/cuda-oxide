# cargo-oxide

Cargo subcommand for building and running Rust GPU programs with cuda-oxide.

Replaces the previous `xtask` pattern with a proper cargo subcommand that works
both inside the cuda-oxide repo (for developers) and externally (for users who
`cargo install`).

## How `cargo oxide build` works

At its core, `cargo oxide` is a thin orchestration layer around `cargo` that
injects the custom codegen backend into the compilation pipeline:

```text
  cargo oxide build vecadd
         |
         v
  1. Discover / build librustc_codegen_cuda.so
         |
         v
  2. Set RUSTFLAGS="-Z codegen-backend=<path>"
         |
         v
  3. cargo build --release
         |
         +-- Host code: rustc compiles normally
         +-- Device code: rustc_codegen_cuda backend:
              - Collect MIR from #[kernel] / #[device] functions
              - Lower to dialect-mir (alloca + mem2reg)
              - Translate to LLVM dialect → textual LLVM IR
              - llc (NVPTX) → PTX
              - (or NVVM IR → libNVVM → LTOIR → nvJitLink → cubin)
         |
         v
  4. Embed device artifact into the host binary
         |
         v
  5. Single binary with host + embedded GPU code
```

The backend is discovered using a priority chain (see **Backend Discovery** below).
Once found, `cargo oxide` sets `RUSTFLAGS` so that every `rustc` invocation in
the build uses `rustc_codegen_cuda` for code generation. The backend handles
both host and device code: host code compiles through the normal LLVM/x86 (or
ARM) path, while `#[kernel]` and `#[device]` functions are extracted, optimized,
and compiled to PTX (or NVVM IR) which is then embedded into the final binary
as a loadable artifact bundle.

## Compilation pipeline: host + device → single binary

### Standard path (PTX)

```text
Rust source
    |
    v
MIR collection (rustc_codegen_cuda backend)
    |
    v
dialect-mir (custom MIR dialect: alloca, load, store, arithmetic)
    |
    v
LLVM dialect (NVPTX-targeted LLVM IR)
    |
    v
textual LLVM IR (.ll file)
    |
    v
llc -mcpu=sm_XX (LLVM with NVPTX backend)
    |
    v
PTX assembly
    |
    v
Embedded into host binary (via custom linker section)
    |
    v
Loaded at runtime with cuda_host::embedded::load_embedded_module
```

### NVVM IR path (kernels with libdevice math)

When a kernel uses Rust float intrinsics (`sin`, `cos`, `exp`, `pow`, ...),
cuda-oxide auto-detects them and emits NVVM IR instead of PTX:

```text
Rust source → MIR → dialect-mir → LLVM dialect → NVVM IR (.ll)
    |
    v
libNVVM (-gen-lto) + libdevice.10.bc
    |
    v
LTOIR
    |
    v
nvJitLink (-arch=sm_XX -lto)
    |
    v
cubin
    |
    v
Loaded at runtime via cuda_host::ltoir::load_kernel_module
```

Both paths produce a loadable module image (PTX or cubin) that the host code
loads at runtime using `cuModuleLoad` or `cuModuleLoadData`.

### Interop workflow

Some examples declare separate device crates that are compiled independently and
loaded by a host program (e.g., a cutile-rs host with a cuda-oxide SIMT device
crate). This is configured via `Cargo.toml` metadata:

```toml
[package.metadata.cuda-oxide]
device-crates = [
    { manifest = "simt/Cargo.toml", ptx-dir = "ptx", artifact-name = "simt_kernels" }
]
```

`cargo oxide run` builds the device crates with the custom backend, writes their
PTX to the configured directory, then builds and runs the host crate normally.

## Installation

**Internal developers** (inside the cuda-oxide repo): no installation needed.
The workspace alias in `.cargo/config.toml` makes `cargo oxide` work immediately.

**External users**:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` automatically fetches and builds the codegen backend
if it's not already available.

## Commands

### `cargo oxide run <example>`

Builds the codegen backend, compiles the example with the custom backend, and
runs it. This is the primary command for day-to-day development.

When neither `--arch` nor `CUDA_OXIDE_TARGET` is set, `run` detects the compute
capability of CUDA device 0 and targets that architecture so the generated PTX
can load on the local GPU. Use `--arch <sm_XXX>` or `CUDA_OXIDE_TARGET=<sm_XXX>`
to override this for a specific device or cross-target workflow.

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol
cargo oxide run device_ffi_test --emit-nvvm-ir --arch sm_120
cargo oxide run cutile_inter_kernel
```

### `cargo oxide build <example>`

Same as `run` but stops after compilation. Useful for examples that require
hardware you don't have (e.g., Blackwell tensor cores).

```bash
cargo oxide build htens          # compiles PTX, doesn't try to run on GPU
cargo oxide build tcgen05        # sm_100a only, but PTX generation works anywhere
```

### `cargo oxide pipeline <example>`

Shows the full compilation pipeline with verbose output at every stage: MIR
collection, `dialect-mir` (pre- and post-`mem2reg`), the LLVM dialect, textual
LLVM IR, and the final PTX.

```bash
cargo oxide pipeline vecadd
cargo oxide pipeline device_ffi_test --emit-nvvm-ir --arch sm_120
```

### `cargo oxide debug <example>`

Builds with debug info (`-C debuginfo=2`) and launches `cuda-gdb`. Supports
`--tui` for GDB's TUI mode and `--cgdb` for the cgdb frontend.

### `cargo oxide new <name> [--async]`

Scaffolds a new standalone cuda-oxide project with `Cargo.toml`,
`rust-toolchain.toml`, and a working `src/main.rs` containing a vector addition
kernel. The default template uses `#[cuda_module]` with typed synchronous launch
methods; `--async` generates a template with `tokio`, `cuda-async`, and typed
lazy `DeviceOperation` launches.

```bash
cargo oxide new my_kernel
cd my_kernel
cargo oxide run
```

### `cargo oxide fmt [--check]`

Formats all crates in the workspace: root workspace, `rustc-codegen-cuda`, and
all examples. With `--check`, reports files that need formatting without
modifying them.

### `cargo oxide doctor`

Validates that your environment is correctly set up. Checks:

- Rust nightly toolchain
- `rust-toolchain.toml`
- Codegen backend `.so`
- CUDA toolkit (`nvcc`)
- libNVVM + nvJitLink + libdevice
- LLVM (`llc` >= 21)

### `cargo oxide setup`

Explicitly builds (or rebuilds) the codegen backend. Normally this happens
automatically on every `run`/`build`/`pipeline` command, but `setup` is useful
after pulling new changes or for CI.

## Configuration and options

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--emit-nvvm-ir` | `run`, `build`, `pipeline` | Generate NVVM IR for libNVVM |
| `--arch <sm_XX>` | `run`, `build`, `pipeline` | Target architecture override |
| `--features <F>` | `run`, `build` | Comma-separated cargo features |
| `-v, --verbose` | `run`, `build` | Show detailed compilation output |
| `--async` | `new` | Use the async template |
| `--cgdb` | `debug` | Use cgdb instead of cuda-gdb |
| `--tui` | `debug` | Use GDB's TUI interface |
| `--check` | `fmt` | Check formatting only |

Environment variables that control the pipeline:

| Variable | Effect |
|----------|--------|
| `CUDA_OXIDE_BACKEND` | Explicit path to `librustc_codegen_cuda.so` |
| `CUDA_OXIDE_TARGET` | GPU architecture (e.g., `sm_120`) |
| `CUDA_OXIDE_LIBDEVICE` | Path to `libdevice.10.bc` |
| `CUDA_OXIDE_LLC` | Path to `llc` binary |
| `CUDA_OXIDE_VERBOSE` | Enable verbose backend output |
| `CUDA_OXIDE_DUMP_MIR` | Dump MIR to files |
| `CUDA_OXIDE_DUMP_LLVM` | Dump LLVM IR to files |
| `CUDA_OXIDE_SHOW_RUSTC_MIR` | Dump rustc's MIR before backend processing |

## Backend discovery

When `cargo oxide` needs the `librustc_codegen_cuda.so` backend, it searches in
this order:

1. **`CUDA_OXIDE_BACKEND` env var** — explicit path override.
2. **Local repo** — detects `crates/rustc-codegen-cuda` relative to workspace
   root, builds from source.
3. **Cached `.so`** — checks `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`,
   but only when it is not older than the running `cargo-oxide` binary.
4. **Auto-fetch** — clones the cuda-oxide repo, builds, and caches (one-time).

Cache staleness (issue #49): `cargo install` rewrites the binary on every
upgrade, bumping its mtime. If the cached `.so` predates the binary, both the
cache and the cached source tree are invalidated so step 4 fetches a fresh copy.

## Integration with Cargo workspace

`cargo oxide` supports two modes:

**Workspace mode** — CWD is inside the cuda-oxide repo (detected by the presence
of `crates/rustc-codegen-cuda`). Examples are resolved from
`crates/rustc-codegen-cuda/examples/`. The backend is built from the local source
tree.

**Standalone mode** — CWD has a `Cargo.toml` but is not inside the workspace.
The backend is located via cache or auto-fetch. Commands like `run` and `build`
operate on the current directory directly, using its `Cargo.toml`.

```toml
# .cargo/config.toml (inside cuda-oxide repo)
[alias]
oxide = "run --package cargo-oxide --"
```

This alias makes `cargo oxide` work for repo developers without installing the
subcommand globally.

## Architecture

```text
crates/cargo-oxide/
├── Cargo.toml
└── src/
    ├── main.rs       # CLI definitions (clap) + dispatch
    ├── backend.rs    # Backend discovery + build logic
    └── commands.rs   # All command implementations
```

## Future commands

| Command | Description |
|---------|-------------|
| `cargo oxide bench <example>` | GPU profiling (nsys/ncu integration), report TFLOPS |
| `cargo oxide test` | Run all examples as a test suite |
| `cargo oxide clean` | Remove generated PTX/LL/LTOIR artifacts and build caches |
| `cargo oxide update` | Update the cached codegen backend to latest version |
| `cargo oxide list` | List examples with descriptions and hardware reqs |
| `cargo oxide inspect <example>` | Show generated PTX without the full pipeline dump |
