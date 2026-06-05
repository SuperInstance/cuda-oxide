# fuzzer

Differential codegen fuzzer for cuda-oxide. This crate finds compiler bugs by
generating random Rust programs, running them through two independent code
generators (the host CPU backend and the cuda-oxide GPU backend), and comparing
the results. Any divergence signals a bug in one of the backends.

## What differential fuzzing means

A compiler backend translates the same intermediate representation (MIR) into
machine code. If two backends are given identical, well-defined input, they
must produce programs that behave identically. Differential fuzzing exploits
this property:

```text
  Random MIR program
         |
         +-----> LLVM/x86 backend -----> CPU trace hash
         |                                    |
         |                                    v
         |                               compare u64
         |                                    |
         |                                    v
         +-----> cuda-oxide/NVPTX backend -> GPU trace hash
```

The CPU result acts as the **oracle**. If the GPU trace differs, the cuda-oxide
backend has a correctness bug. If both crash or miscompile in the same way, the
test is simply discarded. This approach requires no hand-written reference
implementations and no formal specifications — the CPU backend's maturity
provides the ground truth.

### Why it matters for compiler correctness

CUDA GPU compilers are harder to test than CPU compilers:

- **No mature reference:** There is no widely-used alternative Rust-to-PTX
  compiler to compare against.
- **Silent miscompilation is common:** A backend bug may produce wrong PTX that
  still loads and runs without crashing, giving incorrect numerical results.
- **Execution environment is remote:** The GPU has separate memory, different
  rounding behavior for some operations, and divergent control flow. Differential
  testing catches these issues by hashing intermediate values rather than just
  final outputs.

By hashing intermediate values at `dump_var` sites throughout the program, the
fuzzer catches bugs that would be invisible to a simple return-value comparison.

## Architecture

### Three stages

```text
Stage 1: Generate
  rustlantis (vendored MIR generator)
    |
    v
  custom-MIR Rust function + dump_var terminators

Stage 2: Adapt
  mir_generator.py
    |
    v
  Rewrites rustlantis dump calls into fuzzer::dump_var
  Emits generated_case.rs for rustlantis-smoke example

Stage 3: Execute & Compare
  rustlantis-smoke example
    |
    +-- CPU run: host rustc compiles via LLVM, executes on CPU
    +-- GPU run: cuda-oxide backend compiles to PTX, kernel runs on GPU
    |
    v
  Compare FNV-1a trace hashes
```

### Stage 1 — Program generation (rustlantis)

rustlantis is a random MIR program generator. The fuzzer uses a deliberately
small configuration to keep generated programs tractable:

```toml
bb_max_len = 8
max_bb_count = 3
max_bb_count_hard = 6
max_fn_count = 1
max_args_count = 3
var_dump_chance = 1.0
composite_count = 0
adt_count = 0
```

This produces single-function programs with scalar arithmetic, bitwise ops,
comparisons, and control flow — exactly the constructs cuda-oxide's backend
handles today. The seed controls rustlantis' pseudo-random generator, making
failures fully reproducible.

### Stage 2 — Adapter (mir_generator.py)

The Python adapter bridges rustlantis' output to cuda-oxide:

1. **Extract** the first `#[custom_mir]` function from rustlantis' output.
2. **Rewrite** rustlantis' `dump_var(...)` terminators into calls to
   `fuzzer::dump_var(...)`. The adapter injects tuple locals (`__rl_dumpN`)
   so the generated MIR is valid Rust.
3. **Filter** unsupported types. The adapter rejects programs that dump types
   not yet handled by the trace API (e.g., `u128` before the trace API was
   extended). This is `UNSUPPORTED [adapter]` — not a backend bug, just a
   coverage gap.
4. **Emit** `generated_case.rs` with deterministic argument literals and a
   `compute_rustlantis_trace()` wrapper.

### Stage 3 — Execution harness (rustlantis-smoke)

The smoke example is a stable Cargo project that serves as the execution
harness. It contains:

- A hand-written Stage 1b sanity check (`fn0`) to verify the harness itself.
- A `#[kernel]` that calls both `compute_stage1_trace()` and
  `generated_case::compute_rustlantis_trace()`.
- A host driver that runs the kernel, reads back the GPU hashes, and compares
  them to CPU hashes.

The harness is launched via `cargo oxide run rustlantis-smoke`, which compiles
the same source twice: once for the host (LLVM/x86) and once for the device
(cuda-oxide → PTX).

### Trace API

The comparison primitive is a single `u64` global (`RL_TRACE`) updated with
FNV-1a hashing. Every `dump_var` call folds its arguments byte-by-byte into the
trace:

```rust
// In the generated program
__rl_dump0 = (Move(_1), Move(_2), Move(_3));
Call(_5 = dump_var(Move(__rl_dump0)), ...)
```

The `TraceValue` trait covers all scalar types (`bool`, `i8`–`i128`, `u8`–`u128`,
`isize`, `usize`, `char`). `TraceDump` handles tuples up to arity 5. The trace
is `no_std` and uses a `static mut` so the backend cannot constant-fold it away.

## Fuzzer strategy and coverage approach

### What is covered today

| Feature | Status |
|---------|--------|
| Scalar integer arithmetic | ✅ Covered |
| Bitwise ops (`!`, `&`, `\|`, `^`, `<<`, `>>`) | ✅ Covered |
| Comparisons and boolean logic | ✅ Covered |
| Basic blocks and `Goto` | ✅ Covered |
| `Call` / `Return` | ✅ Covered |
| `transmute` | ✅ Covered |
| `dump_var` with tuples | ✅ Covered |
| `u128` / `i128` | ✅ Trace API supports; adapter supports |
| `usize` / `isize` / `char` | ✅ Trace API supports; adapter supports |
| Control flow (switches, loops) | 🔄 Limited by `max_switch_targets=2`, `max_bb_count=6` |
| Composite types (structs, arrays, enums) | ❌ Disabled (`composite_count=0`, `adt_count=0`) |
| Floating point | ❌ Not yet generated by rustlantis config |
| Pointers / references | ❌ Not yet generated by rustlantis config |

### Strategy: small programs, deep coverage

The fuzzer deliberately trades program size for reproducibility and debuggability:

- **Single function:** No cross-function call graph to untangle when debugging.
- **Scalar-only:** No aliasing, no memory layout issues, no drop glue.
- **Deterministic seeds:** Every failure is one `run_seed.py --seed N` away from
  reproduction.
- **Intermediate hashing:** Catches miscompilations even when the final return
  value happens to match by accident.

As cuda-oxide matures, the rustlantis configuration will be widened:
`composite_count`, `adt_count`, `max_fn_count`, and `max_args_count` will
increase to cover structs, arrays, enums, and multi-function programs.

## How to run it

### One seed

```bash
python3 crates/fuzzer/tools/run_seed.py --seed 192
```

### A range of seeds

```bash
python3 crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going --keep-logs
```

- `--keep-going`: Continue after the first non-PASS result.
- `--keep-logs`: Write logs for passing seeds too (normally only failures are
  logged).
- `--no-build`: Skip rebuilding the rustlantis generator (useful when running
  many seeds back-to-back).

### Manually inspect a generated case

```bash
python3 crates/fuzzer/tools/mir_generator.py --seed 192 --function-only
```

This prints the adapted custom-MIR function without the harness wrapper.

## Interpreting results

### Result statuses

| Status | Meaning | Action |
|--------|---------|--------|
| `PASS` | Adapter produced a case, both CPU and GPU ran, trace hashes matched. | None — this is the expected outcome. |
| `MISMATCH` | Both sides ran, but trace hashes differed. | **Investigate immediately.** This is a potential backend correctness bug. |
| `COMPILE_FAIL [backend]` | Adapter produced a case, but cuda-oxide failed to compile or run it. | Check if it's a known limitation (e.g., unsupported MIR construct) or a new crash. |
| `UNSUPPORTED [adapter]` | rustlantis generated a program, but the adapter refused to translate it. | Usually a coverage gap (unsupported type or construct). Widen the adapter or trace API. |

### Example output

```text
seed 0: UNSUPPORTED [adapter] unsupported dumped type for Stage 2 adapter: u128 (...)
seed 1: COMPILE_FAIL [backend] Unsupported construct: Type translation not yet implemented for: RigidTy(Char) (...)
seed 192: PASS

summary: COMPILE_FAIL=1, UNSUPPORTED=1, PASS=1
```

### Artifacts

`run_seed.py` writes to `crates/fuzzer/artifacts/` (gitignored):

```text
artifacts/
├── seed-0-unsupported.log
├── seed-1-compile_fail.log
├── seed-192-pass.log          (only with --keep-logs)
└── summary.jsonl
```

Failure logs include:
- Seed, status, stage, reason, return code
- Full command and command output
- Snapshot of `generated_case.rs` when the adapter produced one

The artifacts directory is cleared at the start of every `run_seed.py`
invocation, so logs always describe only the latest run.

## Relationship to the compiler pipeline

The fuzzer exercises the entire cuda-oxide compilation pipeline end-to-end:

```text
rustlantis custom MIR
         |
         v
  rustc frontend (custom MIR lowering)
         |
         +----> LLVM backend (host) -----> CPU binary
         |
         +----> cuda-oxide backend:
                - MIR collection
                - dialect-mir lowering
                - LLVM dialect → textual LLVM IR
                - llc (NVPTX) → PTX
                - cuModuleLoadData → GPU kernel
```

Because the same `#[kernel]` source is compiled by two different backends from
the same MIR, any divergence points directly to a backend bug. The fuzzer does
not test the Rust frontend (parsing, type checking, borrow checking) — it
assumes rustc's frontend is correct and focuses energy on the codegen backends.

### Integration with CI

The fuzzer is designed for nightly or per-PR regression runs:

```bash
python3 crates/fuzzer/tools/run_seed.py --start 0 --count 100 --keep-going
```

A non-zero exit code indicates at least one `MISMATCH` or `COMPILE_FAIL`, which
should fail the build. `UNSUPPORTED` seeds are expected and do not fail the run.

## Source layout

```text
crates/fuzzer/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Crate root, re-exports trace API
│   └── trace.rs        # FNV-1a trace state, TraceValue / TraceDump traits
├── rustlantis/         # Vendored upstream rustlantis (MIR generator)
└── tools/
    ├── mir_generator.py    # Stage 2 adapter: rustlantis → generated_case.rs
    └── run_seed.py         # Stage 3 driver: generate, build, run, compare
```

The execution harness lives separately at:
`crates/rustc-codegen-cuda/examples/rustlantis-smoke/`
