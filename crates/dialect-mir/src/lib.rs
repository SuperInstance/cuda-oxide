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
//! # Compilation Pipeline
//!
//! ```text
//! Rust source code
//!        │
//!        ▼
//! ┌──────────────┐
//! │   rustc      │  (Stable MIR extraction)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ mir-importer │  (Stable MIR → dialect-mir alloca form)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │   mem2reg    │  (pliron alloca → SSA promotion)
//! └──────┬───────┘
//!        │
//!        ▼
//! dialect-mir (SSA form)      ← THIS DIALECT
//!        │
//!        ▼
//! ┌──────────────┐
//! │  mir-lower   │  (dialect-mir → LLVM dialect)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ llvm-export  │  (LLVM dialect → textual .ll)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │     llc      │  (LLVM IR → PTX)
//! └──────────────┘
//!        │
//!        ▼
//!     *.ptx
//! ```
//!
//! # Passes That Touch This Dialect
//!
//! 1. **Import** (`mir-importer::translate_body`) — Converts rustc Stable MIR
//!    into `dialect-mir` using an alloca + load/store model. Every non-ZST local
//!    becomes a `mir.alloca` slot; definitions emit `mir.store` and uses emit
//!    `mir.load`.
//! 2. **SSA promotion** (`pliron::opts::mem2reg`) — Promotes scalar stack slots
//!    back to SSA values. `MirAllocaOp` implements `PromotableAllocationInterface`
//!    and `MirLoadOp` / `MirStoreOp` implement `PromotableOpInterface` to enable
//!    this pass.
//! 3. **Lowering** (`mir-lower::lower_mir_to_llvm`) — Converts all `dialect-mir`
//!    operations to LLVM dialect operations via pliron's `DialectConversion`
//!    framework.
//!
//! # Modules
//!
//! | Module              | Contents                                       |
//! |---------------------|------------------------------------------------|
//! | [`ops`]             | All dialect operations (54 ops, 11 categories) |
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

/// Dialect attributes (`cast_kind`, `mutability`, `field_index`, `variant_index`, `niche_encoding`).
pub mod attributes;
/// Dialect operations — 54 ops across 11 categories (arithmetic, memory, control flow, …).
pub mod ops;
/// GPU intrinsic op identifiers used by the MIR importer to map rustc calls to NVVM ops.
pub mod rust_intrinsics;
/// Dialect types (`ptr`, `slice`, `struct`, `enum`, `array`, `tuple`, `f16`).
pub mod types;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

/// Name of the MIR dialect in pliron (`"mir"`).
pub const MIR_DIALECT_NAME: &str = "mir";

/// Register all dialect ops, types, and attributes with the given context.
pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::new(MIR_DIALECT_NAME));
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
