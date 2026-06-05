# dialect-nvvm

A [pliron](https://github.com/vaivaswatha/pliron) dialect that models NVIDIA GPU
intrinsics as typed IR operations. It is the bridge between Rust-level device
abstractions (`cuda_device::thread`, `cuda_device::tma`, …) and the LLVM NVPTX
backend that ultimately emits PTX assembly.

## What is NVVM?

NVVM (NVIDIA Virtual Machine) IR is NVIDIA's LLVM-IR-compatible representation
for GPU kernels. It looks like standard LLVM IR — SSA values, basic blocks,
`load`/`store` — but adds a namespace of GPU-specific intrinsics under
`llvm.nvvm.*`. These intrinsics expose thread indexing, warp collectives,
barriers, tensor-memory operations, and other hardware features that have no
equivalent in CPU LLVM.

In the cuda-oxide pipeline the flow is:

```text
Rust source ──▶ rustc MIR ──▶ dialect-mir ──▶ dialect-nvvm ──▶ LLVM IR ──▶ PTX
                                                    ↑
                                              THIS CRATE
```

`mir-lower` generates `dialect-nvvm` ops when it recognises calls to
`cuda_device` intrinsics. `llvm-export` then emits each op as either an
`@llvm.nvvm.*` intrinsic call or an inline PTX `asm` fragment, depending on
whether LLVM's NVPTX backend already has a first-class intrinsic for the
operation.

## Operations and GPU Semantics

The dialect contains **133 public items** organised by functional area and
minimum GPU architecture. Every op is a zero-cost abstraction: it carries no
runtime overhead beyond the instruction the hardware executes.

### Universal (all GPUs)

| Module | Ops | Semantics |
|--------|-----|-----------|
| `thread` | 18 | Read PTX special registers (`%tid.x`, `%ctaid.x`, `%ntid.x`, …), block-wide barrier (`bar.sync 0`), memory fences (`membar.cta` / `gl` / `sys`) |
| `warp` | 18 | Shuffle (`shfl.idx`, `shfl.bfly`, `shfl.down`, `shfl.up`), vote (`vote.all`, `vote.any`, `vote.ballot`), warp-match (`match.any`, `match.all`), lane-id query |
| `grid` | 1 | Cooperative grid-wide sync (`grid.sync`) for multi-block cooperative launches |
| `debug` | 7 | `clock`, `clock64`, `globaltimer`, `trap`, `breakpoint`, `pm_event`, `vprintf` |

### Volta+ (sm_70)

| Module | Ops | Semantics |
|--------|-----|-----------|
| `atomic` | 4 | Atomic load, store, RMW and compare-and-swap with explicit `AtomicOrdering` (relaxed … seq_cst) and `AtomicScope` (cta / gpu / sys) |

### Hopper+ (sm_90)

| Module | Ops | Semantics |
|--------|-----|-----------|
| `cluster` | 11 | Thread-block-cluster CTA IDs, cluster sync, distributed shared memory (`mapa_shared_cluster`, `dsmem_read`) |
| `mbarrier` | 10 | Async hardware barriers: init, arrive, arrive-expect-tx, try-wait, wait-parity, fence-proxy-async |
| `tma` | 15 | Tensor Memory Accelerator copies: 1-D through 5-D global↔shared, multicast, CTA-group-2 variants, commit/wait groups |
| `wgmma` | 5 | Warpgroup MMA: fence, commit, wait, shared-memory descriptor build, M64×N64×K16 `bf16` MMA |
| `stmatrix` | 5 | Shared-memory matrix stores: `m8n8` ×2, ×4, transposed, `f32`→`bf16` convert |

### Blackwell+ (sm_100)

| Module | Ops | Semantics |
|--------|-----|-----------|
| `tcgen05` | 24 | 5th-generation tensor cores + TMEM: alloc/dealloc, fences, MMA (`f16`/`bf16`/`tf32`, warpgroup and non-warpgroup), SMEM↔TMEM copies, pure loads, CTA-pair (`cg2`) variants |
| `clc` | 6 | Cluster Launch Control: try-cancel, query-is-canceled, query first-ctaid per dimension |

## How NVVM Maps to PTX

Each `dialect-nvvm` op lowers through **one of two paths**:

### 1. LLVM NVVM Intrinsic Call

Simple ops that have a direct LLVM intrinsic are emitted as `llvm.call`:

```llvm
; dialect-nvvm: ReadPtxSregTidXOp
%tid.x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()

; dialect-nvvm: Barrier0Op
call void @llvm.nvvm.barrier0()
```

LLVM's NVPTX backend recognises these intrinsics and translates them to the
corresponding PTX instructions during code generation:

```ptx
mov.u32     %r1, %tid.x;
bar.sync    0;
```

### 2. Inline PTX Assembly

Complex ops — especially those introduced in recent GPU generations that LLVM
does not yet expose as intrinsics — are emitted as inline `asm` fragments:

```llvm
; wgmma.mma_async (Hopper)
call void asm sideeffect "wgmma.mma_async ...", "..."(...)

; tcgen05.mma (Blackwell)
call void asm sideeffect "tcgen05.mma ...", "..."(...)
```

This gives cuda-oxide access to bleeding-edge hardware instructions without
waiting for upstream LLVM support. The inline-asm path is used for `wgmma`,
`tcgen05`, and some TMA variants.

## Intrinsic Functions Exposed

The crate exposes its ops through a single `register` entry point:

```rust
use pliron::context::Context;
use dialect_nvvm::register;

let mut ctx = Context::new();
register(&mut ctx);   // registers the "nvvm" dialect and all ops
```

Each op implements pliron's `Op` and `Verify` traits. Verification is
**structural, not semantic**: we check operand/result counts and (for some
ops) that results are `i32`, but we deliberately do not replicate LLVM's type
system. The ops are machine-generated by `mir-lower`, not written by hand, so
type errors are caught by rustc long before they reach this dialect. LLVM
itself validates the intrinsic types when the `.ll` file is processed.

## Source Layout

```text
src/
├── lib.rs          # Dialect registration
└── ops/
    ├── mod.rs       # Op registry + architecture table
    ├── thread.rs    # Thread/block indexing, barrier0, threadfences
    ├── warp.rs      # Shuffle, vote, match operations
    ├── grid.rs      # Cooperative grid_sync
    ├── atomic.rs    # Atomic ops + ordering/scope/kind attributes
    ├── debug.rs     # Clock/timer, trap, printf
    ├── cluster.rs   # Thread block clusters, DSMEM
    ├── mbarrier.rs  # Async barriers for TMA
    ├── tma.rs       # Tensor Memory Accelerator copies
    ├── wgmma.rs     # Warpgroup MMA (Hopper)
    ├── stmatrix.rs  # Shared memory matrix stores
    ├── tcgen05.rs   # 5th-gen tensor cores + TMEM (Blackwell)
    └── clc.rs       # Cluster Launch Control (Blackwell)
```

## Further Reading

- [`dialect-mir`](../dialect-mir/) — pliron dialect modelling Rust MIR (lowering source)
- [`mir-lower`](../mir-lower/) — generates `dialect-nvvm` ops during lowering
- [`llvm-export`](../llvm-export/) — pliron-llvm shim + textual `.ll` exporter (lowering target)
- [`cuda-device`](../cuda-device/) — user-facing intrinsics that map to `dialect-nvvm` ops
