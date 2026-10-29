import bench.tests.IntegerLiteralAssignmentBody
namespace VeriTile.Tests.TypedIntegerOperandBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  typedFP32Storage := Bool.true
/-- Event sentinels distinguish operand order, integer channel and precision. -/
def model : Algebra Int :=
  { IntegerLiteralAssignmentBody.model with
    binary := fun p _ _ a b => if p = some .fp32 then 100*a+b else 5000 }
def initial : State Int :=
  { IntegerLiteralAssignmentBody.initial with
    mem := fun _ _ => .mk .fp32 10
    pids := fun _ => 3 }

def rightInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.full((), 2, tl.int64)
    result = x / n
    tl.store(Y, result)
  }
def leftInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.full((), 2, tl.int64)
    result = n / x
    tl.store(Y, result)
  }
def naturalInteger (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    n = tl.program_id(0)
    result = x / n
    tl.store(Y, result)
  }

def reducedProduct (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, 1)
    x = tl.load(X + offsets)
    wdy = x * x
    n = tl.full((), 2, tl.int64)
    result = tl.sum(wdy, axis=0) / n
    tl.store(Y, result)
  }

/-- The reduction sentinel distinguishes an actual typed FP32 reduction from
an erased precision annotation; the result still divides by a signed tensor. -/
def reductionModel : Algebra Int :=
  { model with
    reduceSumTyped := fun p dtype {_} _ _ _ => some (fun _ =>
      ofFloat dtype (if p = some .fp32 ∧ dtype = .fp32 then (11 : Int) else 9000)) }
theorem product_reduction_keeps_fp32_conversion :
    (FP.Structural.exec reductionModel (reducedProduct "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .fp32 1102) := by
  simp [FP.Structural.exec,reducedProduct,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,reductionModel,model,initial,
    IntegerLiteralAssignmentBody.model,IntegerLiteralAssignmentBody.initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean product_reduction_keeps_fp32_conversion

theorem signed_right_keeps_fp32_conversion :
    (FP.Structural.exec model (rightInteger "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .fp32 1002) := by
  simp [FP.Structural.exec,rightInteger,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.model,IntegerLiteralAssignmentBody.initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean signed_right_keeps_fp32_conversion

theorem signed_left_keeps_operand_order :
    (FP.Structural.exec model (leftInteger "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .fp32 210) := by
  simp [FP.Structural.exec,leftInteger,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.model,IntegerLiteralAssignmentBody.initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean signed_left_keeps_operand_order

theorem natural_uses_unsigned_conversion :
    (FP.Structural.exec model (naturalInteger "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .fp32 997) := by
  simp [FP.Structural.exec,naturalInteger,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.model,IntegerLiteralAssignmentBody.initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean natural_uses_unsigned_conversion

def dtypeSignature : CallSignature := { signature with constexpr := ["dtype"] }
def sourceDType (X Y : RegionName) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  with_triton_signature dtypeSignature in triton {
    x = tl.load(X)
    tl.store(Y, x.to(dtype))
  }
theorem source_dtype_matches_caller :
    (FP.Structural.exec model (sourceDType "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .fp32 10) := by
  simp [FP.Structural.exec,sourceDType,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.model,IntegerLiteralAssignmentBody.initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean source_dtype_matches_caller

/-- error: typed FP32 storage requires its dtype constexpr to be the closed fp32 caller value -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  let dtype : FloatDType := .fp16
  with_triton_signature dtypeSignature in triton {
    tl.store(Y, tl.load(X).to(dtype))
  }

/-- error: dtype constexpr mutation is unsupported -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  with_triton_signature dtypeSignature in triton {
    dtype = tl.float16
    tl.store(Y, tl.load(X).to(dtype))
  }

end
end VeriTile.Tests.TypedIntegerOperandBody
