/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect — Rust-semantic Pliron IR dialect for cuda-oxide.
//!
//! `dialect-mir` models Rust's Mid-level IR (MIR) as a Pliron dialect,
//! preserving Rust-specific type semantics (enums, slices, pointers with
//! address spaces, structs with `#[repr(Rust)]` layouts) before lowering
//! to the LLVM dialect via `mir-lower`.
//!
//! # Dialect Position in the Pipeline
//!
//! ```text
//! rustc MIR (Stable MIR)
//!       │  ← mir-importer translates here
//!       ▼
//! dialect-mir (alloca form)   ← THIS DIALECT
//!       │  ← pliron mem2reg promotes allocas → SSA
//!       ▼
//! dialect-mir (SSA form)
//!       │  ← mir-lower lowers to LLVM dialect
//!       ▼
//! LLVM dialect (pliron-llvm)
//! ```
//!
//! # Modules
//!
//! | Module              | Contents                                       |
//! |---------------------|------------------------------------------------|
//! | [`ops`]             | All dialect operations (44 ops, 11 categories) |
//! | [`types`]           | Dialect types (ptr, slice, struct, enum, …)    |
//! | [`attributes`]      | Dialect attributes (cast kind, etc.)           |
//! | [`rust_intrinsics`] | GPU intrinsic op identifiers                   |
//!
//! # Type System
//!
//! The dialect extends pliron's builtin integer/float types with:
//! - [`types::MirPtrType`] — typed pointer with address-space tracking (generic/global/shared/…)
//! - [`types::MirSliceType`] — fat pointer (`ptr` + `usize` len)
//! - [`types::MirStructType`] — struct with rustc field offsets for ABI correctness
//! - [`types::MirEnumType`] — discriminant + payload union
//! - [`types::MirArrayType`] — `[T; N]`
//! - [`types::MirTupleType`] — heterogeneous tuple
//! - [`types::MirFP16Type`] — IEEE-754 `f16`

pub mod attributes;
pub mod ops;
pub mod rust_intrinsics;
pub mod types;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const MIR_DIALECT_NAME: &str = "mir";

pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::new(MIR_DIALECT_NAME));
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
