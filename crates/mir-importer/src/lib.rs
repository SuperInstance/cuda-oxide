/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIR translation functions often have many parameters to pass context
#![allow(clippy::too_many_arguments)]
// Complex types are unavoidable when working with rustc internals
#![allow(clippy::type_complexity)]

//! Rust MIR to `dialect-mir` translator and compilation pipeline for cuda-oxide.
//!
//! This crate is cuda-oxide's compiler **frontend**. It consumes Rust's
//! Stable MIR (the Mid-level IR produced by `rustc`) and emits
//! [`dialect-mir`][dialect_mir] — a pliron dialect that preserves Rust
//! semantics — then drives the rest of the GPU compilation pipeline down to
//! PTX.
//!
//! # What is MIR?
//!
//! **MIR** (Mid-level Intermediate Representation) is the IR `rustc` uses
//! after type-checking and before LLVM IR. A MIR function body consists of:
//!
//! | Concept | Description |
//! |---------|-------------|
//! | **Locals** | Typed variables `_0`, `_1`, `_2`, … (`_0` is the return value) |
//! | **Basic blocks** | Sequences of **statements** ending in a **terminator** |
//! | **Statements** | Simple operations: assignments, storage markers |
//! | **Terminators** | Control flow: `goto`, `call`, `return`, `switchInt` |
//! | **Rvalues** | Right-hand sides: `BinaryOp`, `Ref`, `Use`, `Aggregate`, … |
//! | **Places** | Lvalues: a local plus zero or more *projections* |
//!
//! MIR is **not** SSA. A local can be assigned in `bb0` and read in `bb1`.
//! The translator bridges this gap with the [alloca + load/store model](#alloca--loadstore-model).
//!
//! # What is Pliron?
//!
//! **Pliron** is an MLIR-like compiler framework in Rust. It provides:
//!
//! - **Operations** — structured instructions that can contain *regions*,
//!   which contain *blocks*, which contain more ops.
//! - **Dialects** — namespaces of related ops and types. `dialect-mir` models
//!   Rust MIR; `dialect-nvvm` models CUDA intrinsics; the LLVM dialect models
//!   LLVM IR.
//! - **Types** — integers, floats, structs, arrays, pointers with address
//!   spaces, and user-defined dialect types.
//! - **Passes** — transformations like `mem2reg` (promote allocas to SSA) and
//!   `DialectConversion` (lower one dialect to another).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────── mir-importer ──────────────────────────────────┐
//! │                                                                       │
//! │  ┌──────────────┐   ┌─────────────────────────────────────────────┐   │
//! │  │  translator  │──▶│                  pipeline                   │   │
//! │  │              │   │                                             │   │
//! │  │     MIR      │   │  dialect-mir (alloca)                       │   │
//! │  │      ──▶     │   │    ──▶ mem2reg                              │   │
//! │  │  dialect-mir │   │    ──▶ dialect-mir (SSA)                    │   │
//! │  │   (alloca)   │   │    ──▶ LLVM dialect  (via mir-lower)        │   │
//! │  │              │   │    ──▶ LLVM IR ──▶ PTX  (via llc)           │   │
//! │  └──────────────┘   └─────────────────────────────────────────────┘   │
//! │                                                                       │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Pipeline Steps
//!
//! 1. **Translate** — `translator::body::translate_body` converts each MIR
//!    function into a `mir.func` inside a `builtin.module`.
//! 2. **Verify** — Check type consistency and structural invariants on the
//!    `dialect-mir` module.
//! 3. **mem2reg** — Promote scalar alloca slots back to SSA via
//!    [`pliron::opts::mem2reg`][mem2reg].
//! 4. **Lower** — Convert `dialect-mir` → LLVM dialect (via [`mir-lower`]).
//! 5. **Verify** — Check the LLVM dialect module.
//! 6. **Export** — Write textual LLVM IR (`.ll`) via [`llvm-export`].
//! 7. **Generate PTX** — Invoke `llc` for PTX assembly.
//!
//! # Key Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`translator::body`](translator::body) | Function-level translation, alloca setup, kernel attributes |
//! | [`translator::block`](translator::block) | Basic block translation coordinator |
//! | [`translator::statement`](translator::statement) | Statement translation (assignments, projections, storage markers) |
//! | [`translator::terminator`](translator::terminator) | Terminator translation (goto, call, return, switch, assert) |
//! | [`translator::rvalue`](translator::rvalue) | Expression translation (binops, casts, aggregates, constants) |
//! | [`translator::types`](translator::types) | Rust type → `dialect-mir` type conversion |
//! | [`translator::values`](translator::values) | MIR local → alloca-slot mapping + address-space inference |
//! | [`pipeline`](pipeline) | `mem2reg`, lower to LLVM dialect, export LLVM IR, run `llc` |
//! | [`error`](error) | Error types integrated with pliron's error system |
//!
//! # Alloca + Load/Store Model
//!
//! Every non-ZST MIR local is materialised as a single `mir.alloca` emitted
//! at the top of the function's entry block. Defs lower to `mir.store`, uses
//! lower to `mir.load`. Cross-block data flow happens through the slots, so
//! blocks (other than the entry) take no arguments.
//!
//! ```text
//! Rust MIR (not strict SSA):           dialect-mir (alloca + load/store):
//!
//! bb0: {                               ^bb0(%arg0: i32, ...):
//!     _1 = 42;                             %s1 = mir.alloca : !mir.ptr<i32>
//!     goto -> bb1;                         %v1 = mir.const 42 : i32
//! }                                        mir.store %v1, %s1
//! bb1: {                                   mir.goto ^bb1
//!     _2 = _1;   // cross-block read!  ^bb1:
//!     return;                              %r = mir.load %s1
//! }                                        mir.return %r : i32
//! ```
//!
//! Pliron's `mem2reg` pass promotes the slots back into SSA before the
//! `dialect-mir` → LLVM dialect lowering runs.
//!
//! # Address-Space Inference
//!
//! Rust pointer types carry no address-space information, but GPU locals often
//! hold pointers in concrete address spaces (e.g. `addrspace(3)` for shared
//! memory). [`translator::values::SlotAddrSpaceMap`](translator::values) pre-
//! scans the MIR body and infers the address space for each local's alloca
//! slot from the writes into it, avoiding expensive `addrspacecast` round-
//! trips at every load/store.
//!
//! # Example
//!
//! ```rust,ignore
//! use pliron::context::Context;
//! use rustc_public::mir::mono::Instance;
//!
//! // Inside rustc callback:
//! let body = instance.body().unwrap();
//! let mut ctx = Context::new();
//!
//! let module_op = mir_importer::translator::translate_function(
//!     &mut ctx, &body, &instance, /* is_kernel */ true
//! )?;
//! ```
//!
//! [dialect_mir]: ../../dialect_mir/
//! [mem2reg]: pliron::opts::mem2reg
//! [mir-lower]: ../../mir-lower/
//! [llvm-export]: ../../llvm-export/

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_span;

pub mod error;
pub mod pipeline;
pub mod translator;

pub use error::{TranslationErr, TranslationResult};
pub use pipeline::{
    CollectedFunction, CompilationArtifactKind, CompilationResult, DeviceExternAttrs,
    DeviceExternDecl, PipelineConfig, PipelineError, run_pipeline,
};
