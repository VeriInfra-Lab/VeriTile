import VeriTile.Triton.DSL.SignedRankOneBlockPointers
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SignedRankOneBlockPointers
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def identityHelper : PureJitFunction := triton_function (x) {
    return x
}
def signature : CallSignature where
  pointers := [("Input",.fp32),("Output",.fp32)]
  scalars := [("stride",.integer),("offset",.integer),("parent",.integer)]
  constexpr := ["BLOCK","order0"]
  readPorts := ["Input"]
  writePorts := ["Output"]
def source (Input Output : RegionName) (input_base stride offset : Int) (parent BLOCK : Nat) : ComputeKernel :=
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := stride,offset := offset,parent := parent]
    signed_pointer_offsets [Input := input_base,Output := 0]
    pure_functions [copy_value := identityHelper] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(order0,)).to(Input.type.element_ty)
      copied = copy_value(value)
      output_block = tl.make_block_ptr(Output, (parent,), (1,), (offset,), (BLOCK,), order=(order0,))
      tl.store(output_block, copied.to(output_block.type.element_ty), boundary_check=(order0,))
    }

def floatingSignature : CallSignature := { signature with
  pointers := [], floatingPointers := [("Input",.fp32),("Output",.fp32)] }
def floatingSource (Input Output : RegionName) (input_base stride offset : Int) (parent BLOCK : Nat) : ComputeKernel :=
  let order0 : Nat := 0
  with_triton_call floatingSignature call_args [stride := stride,offset := offset,parent := parent]
    signed_pointer_offsets [Input := input_base,Output := 0]
    pure_functions [copy_value := identityHelper] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(order0,)).to(Input.type.element_ty)
      copied = copy_value(value)
      output_block = tl.make_block_ptr(Output, (parent,), (1,), (offset,), (BLOCK,), order=(order0,))
      tl.store(output_block, copied.to(output_block.type.element_ty), boundary_check=(order0,))
    }

def zeroHelper : PureJitFunction := triton_function (x) {
    return 0
}
def integerOutputSignature : CallSignature := { floatingSignature with
  pointers := [("Output",.i64)], floatingPointers := [("Input",.fp32)] }
def integerOutputSource (Input Output : RegionName) (input_base stride offset : Int) (parent BLOCK : Nat) : ComputeKernel :=
  let order0 : Nat := 0
  with_triton_call integerOutputSignature call_args [stride := stride,offset := offset,parent := parent]
    signed_pointer_offsets [Input := input_base,Output := 0]
    pure_functions [copy_value := zeroHelper] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(order0,)).to(Input.type.element_ty)
      copied = copy_value(value)
      output_block = tl.make_block_ptr(Output, (parent,), (1,), (offset,), (BLOCK,), order=(order0,))
      tl.store(output_block, copied.to(output_block.type.element_ty), boundary_check=(order0,))
    }

def converted {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (address : Nat) : Cell α :=
  .mk .real (M.cast (some .fp32) .real .real
    (M.cast (some .fp32) .real .real (M.fp32Load ((s.mem "Input" address).read .real))))

def convertedFloating {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (address : Nat) : Cell α :=
  .mk .fp32 (M.cast none .fp32 .fp32 (M.cast none .fp32 .fp32 ((s.mem "Input" address).read .fp32)))

theorem compensated_negative_base {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" (-2) 1 2 4 1) s = some t ∧
      t.mem = (s.write "Output" 2 (converted M s 0)).mem := by
  simp [source,converted,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem negative_stride {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 3 (-1) 0 4 4) s = some t ∧
      t.mem = ((((s.write "Output" 0 (converted M s 3)).write "Output" 1 (converted M s 2)).write "Output" 2 (converted M s 1)).write "Output" 3 (converted M s 0)).mem := by
  simp [source,converted,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem zero_stride {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 5 0 0 2 2) s = some t ∧
      t.mem = ((s.write "Output" 0 (converted M s 5)).write "Output" 1 (converted M s 5)).mem := by
  simp [source,converted,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem upper_boundary {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 0 1 2 3 4) s = some t ∧
      t.mem = (s.write "Output" 2 (converted M s 2)).mem := by
  simp [source,converted,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem lower_boundary_empty {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 8 1 (-4) 3 4) s = some t ∧
      t.mem = (s).mem := by
  simp [source,converted,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem floating_storage_casts {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (floatingSource "Input" "Output" (-2) 1 2 4 1) s = some t ∧
      t.mem = (s.write "Output" 2 (convertedFloating M s 0)).mem := by
  simp [floatingSource,convertedFloating,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem integer_return_keeps_its_type {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (integerOutputSource "Input" "Output" (-2) 1 2 4 1) s = some t ∧
      t.mem = (s.write "Output" 2 (.mk .int 0)).mem := by
  simp [integerOutputSource,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,bop,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    State.setReg,State.write,TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem inactive_zero_capacity {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    Safety.traceSafeList M (fun _ => 0) (source "Input" "Output" 8 1 (-1) 3 1).surfaceBody s := by
  simp only [source,ComputeKernel.surfaceBody]
  simp [exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.addressSafe,Safety.safeAddress_unfold,Safety.safeMask,Safety.maskActive,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

theorem active_zero_capacity_rejected {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ¬ Safety.traceSafeList M (fun _ => 0) (source "Input" "Output" 0 1 0 1 1).surfaceBody s := by
  simp only [source,ComputeKernel.surfaceBody]
  simp [exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.addressSafe,Safety.safeAddress_unfold,Safety.safeMask,Safety.maskActive,
    ComputeDType.eraseDType,Option.bind,evalOffsets,store,numeric,numericLt,numericLe,numericEq,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.allIndices,List.finRange_succ,Region.cast,ofFloat,toFloat]

/-- error: rank-one block order must be an immutable caller constexpr zero -/
#guard_msgs in
def wrong_order (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 1
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
    }

/-- error: rank-one block order must be an immutable caller constexpr zero -/
#guard_msgs in
def runtime_order (Input Output : RegionName) (dynamic_order : Nat) : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := dynamic_order
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
    }

/-- error: rank-one boundary check must select axis zero -/
#guard_msgs in
def wrong_boundary (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(1,))
    }

/-- error: rank-one block base needs an explicit signed caller offset -/
#guard_msgs in
def missing_base (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
    }

/-- error: queried block-pointer parameters must remain immutable -/
#guard_msgs in
def pointer_rebound (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      Input = Output
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
    }

/-- error: queried block-pointer parameters must remain immutable -/
#guard_msgs in
def order_rebound (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      order0 = 1
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
    }

/-- error: rank-one block pointers require supported boundary-checked load or store -/
#guard_msgs in
def block_alias (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      aliased = input_block
    }

/-- error: pointer dtype cast requires known caller storage metadata -/
#guard_msgs in
def unknown_dtype (Input Output : RegionName)  : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(order0,)).to(Other.type.element_ty)
    }

/-- error: pure JIT helper cannot reference free names or other functions -/
#guard_msgs in
def forged_helper (Input Output : RegionName) : ComputeKernel :=
  let BLOCK : Nat := 1
  let order0 : Nat := 0
  let bad : PureJitFunction := ⟨"x","return missing"⟩
  with_triton_call signature call_args [stride := 1,offset := 0,parent := 4]
    signed_pointer_offsets [Input := 0,Output := 0] pure_functions [copy_value := bad] in triton {
      input_block = tl.make_block_ptr(Input, (parent,), (stride,), (offset,), (BLOCK,), order=(order0,))
      value = tl.load(input_block, boundary_check=(order0,))
      copied = copy_value(value)
    }

#axiomsClean integer_return_keeps_its_type
#axiomsClean inactive_zero_capacity
#axiomsClean active_zero_capacity_rejected
#axiomsClean floating_storage_casts
#axiomsClean compensated_negative_base
#axiomsClean negative_stride
#axiomsClean zero_stride
#axiomsClean upper_boundary
#axiomsClean lower_boundary_empty
end
end VeriTile.Tests.SignedRankOneBlockPointers
