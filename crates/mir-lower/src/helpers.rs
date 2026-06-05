/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helper functions for `dialect-mir` → LLVM dialect lowering.
//!
//! Shared utilities across operation converters: intrinsic declaration and
//! IR hierarchy navigation. For constant creation, use
//! [`crate::convert::intrinsics::common`] (`create_i{n}_const` variants that
//! take a `DialectConversionRewriter`).
//!
//! # Functions
//!
//! ## Intrinsic Declaration
//!
//! | Function                          | Description                              |
//! |-----------------------------------|------------------------------------------|
//! | [`ensure_intrinsic_declared`]     | Declare intrinsic in module              |
//!
//! ## IR Navigation
//!
//! | Function                  | Description                          |
//! |---------------------------|--------------------------------------|
//! | [`get_parent_func`]       | Get the function containing a block  |
//! | [`get_module_from_block`] | Get the module containing a block    |
//!
//! # Usage Pattern
//!
//! These helpers are typically called from operation converters:
//!
//! ```ignore
//! // In an intrinsic converter:
//! fn convert_thread_id(ctx: &mut Context, block: Ptr<BasicBlock>) -> Result<()> {
//!     let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
//!     let func_ty = FuncType::get(ctx, i32_ty.into(), vec![], false);
//!     ensure_intrinsic_declared(ctx, block, "llvm_nvvm_read_ptx_sreg_tid_x", func_ty)?;
//!     // ...
//! }
//! ```

use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;

// ============================================================================
// Intrinsic Declaration
// ============================================================================

/// Ensure an intrinsic function is declared in the module.
///
/// LLVM intrinsics must be declared as function symbols in the module before
/// they can be called. This function checks if the intrinsic already exists
/// and creates a declaration if it doesn't.
///
/// # Arguments
///
/// * `ctx` - The pliron context for IR manipulation
/// * `llvm_block` - A block in the function where we need the intrinsic
///   (used to navigate to the parent module)
/// * `intrinsic_name` - The name of the intrinsic (e.g., `"llvm_nvvm_read_ptx_sreg_tid_x"`)
/// * `func_ty` - The function type of the intrinsic
///
/// # Returns
///
/// `Ok(())` if the intrinsic was already declared or was successfully created,
/// or an error if the IR structure is invalid.
///
/// # IR Navigation
///
/// The function navigates the IR hierarchy as follows:
///
/// ```text
/// llvm_block → parent_func → module → module_block
///                                          ↓
///                                   insert func declaration here
/// ```
///
/// # Example
///
/// ```ignore
/// // Declare the thread ID intrinsic
/// let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
/// let func_ty = FuncType::get(ctx, i32_ty.into(), vec![], false);
/// ensure_intrinsic_declared(ctx, llvm_block, "llvm_nvvm_read_ptx_sreg_tid_x", func_ty)?;
///
/// // Now we can call the intrinsic
/// let call_op = llvm::CallOp::new(ctx, "llvm_nvvm_read_ptx_sreg_tid_x", vec![], ...);
/// ```
///
/// # Note on Intrinsic Naming
///
/// LLVM NVVM intrinsics use underscores in their names when represented in the
/// Pliron IR symbol system (e.g., `llvm_nvvm_read_ptx_sreg_tid_x`). These are
/// translated to the standard LLVM intrinsic names with dots when exported to
/// LLVM IR (e.g., `llvm.nvvm.read.ptx.sreg.tid.x`).
pub fn ensure_intrinsic_declared(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    intrinsic_name: &str,
    func_ty: pliron::r#type::TypePtr<llvm_export::types::FuncType>,
) -> Result<(), anyhow::Error> {
    // Navigate from block to parent function
    let func_op = llvm_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Block has no parent operation (expected function)"))?;

    // Navigate from function to parent module
    let module_op = func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Function has no parent operation (expected module)"))?;

    // Get the module's single region and its entry block
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow::anyhow!("Module region is empty (no entry block)"))?;

    // Convert intrinsic name to identifier
    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .try_into()
        .map_err(|e| anyhow::anyhow!("Invalid intrinsic name '{}': {:?}", intrinsic_name, e))?;

    // Check if the intrinsic is already declared
    let mut already_declared = false;
    for existing_op in module_block.deref(ctx).iter(ctx) {
        if let Some(existing_func) =
            pliron::operation::Operation::get_op::<llvm::FuncOp>(existing_op, ctx)
            && existing_func.get_symbol_name(ctx) == sym_name
        {
            already_declared = true;
            break;
        }
    }

    // If not declared, create a function declaration (no body)
    if !already_declared {
        let func_decl = llvm::FuncOp::new(ctx, sym_name, func_ty);
        // Insert before the current function (keeps intrinsics at top of module)
        func_decl.get_operation().insert_before(ctx, func_op);
    }

    Ok(())
}

// ============================================================================
// IR Navigation
// ============================================================================

/// Get the parent function operation from a basic block.
///
/// In the pliron IR hierarchy, a basic block is contained in a region,
/// which is contained in an operation (typically a function).
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `llvm_block` - The basic block to find the parent of
///
/// # Returns
///
/// `Ok(Ptr<Operation>)` pointing to the parent function operation,
/// or an error if the block has no parent.
///
/// # IR Hierarchy
///
/// ```text
/// ModuleOp
/// └── Region
///     └── Block
///         └── FuncOp          ← returned by this function
///             └── Region
///                 ├── Block   ← llvm_block parameter
///                 ├── Block
///                 └── ...
/// ```
pub fn get_parent_func(
    ctx: &Context,
    llvm_block: Ptr<BasicBlock>,
) -> Result<Ptr<pliron::operation::Operation>, anyhow::Error> {
    llvm_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Block has no parent operation"))
}

/// Get the module operation from a basic block.
///
/// Navigates two levels up in the IR hierarchy: from block to function,
/// then from function to module.
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `llvm_block` - A basic block within a function within the module
///
/// # Returns
///
/// `Ok(Ptr<Operation>)` pointing to the module operation,
/// or an error if the hierarchy is incomplete.
///
/// # IR Hierarchy
///
/// ```text
/// ModuleOp                    ← returned by this function
/// └── Region
///     └── Block
///         ├── FuncOp
///         │   └── Region
///         │       └── Block   ← llvm_block parameter
///         └── FuncOp
///             └── ...
/// ```
///
/// # Example
///
/// ```ignore
/// // Find the module to add a global declaration
/// let module_op = get_module_from_block(ctx, conv_ctx.llvm_block)?;
/// let module_region = module_op.deref(ctx).get_region(0);
/// let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
/// // Now we can insert global declarations in module_block
/// ```
pub fn get_module_from_block(
    ctx: &Context,
    llvm_block: Ptr<BasicBlock>,
) -> Result<Ptr<pliron::operation::Operation>, anyhow::Error> {
    let func_op = get_parent_func(ctx, llvm_block)?;
    func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Function has no parent operation (expected module)"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Note: Testing these helpers requires a full pliron context with
    // dialects registered, which is complex to set up. Integration tests
    // in the `tests/` directory cover these functions indirectly.
}
