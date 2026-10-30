import bench.tests.IntegerLiteralAssignmentBody
namespace VeriTile.Tests.KnownIntegerRangeStepBody
open VeriTile.Triton FP.Structural
open IntegerLiteralAssignmentBody (model)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Y",.i64)]
def initial : State Int :=
  { IntegerLiteralAssignmentBody.initial with mem := fun _ _ => .mk .int 0 }

def captured (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    one = tl.full((), 1, tl.int64)
    total = tl.full((), 0, tl.int64)
    for i in range(0, 3, one):
        total += 1
        one = tl.full((), 9, tl.int64)
    tl.store(Y, total)
  }
def reassigned (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    one = tl.full((), 1, tl.int64)
    one = tl.full((), 2, tl.int64)
    total = tl.full((), 0, tl.int64)
    for i in range(0, 5, one):
        total += 1
    tl.store(Y, total)
  }
def descending (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    stride = -1
    total = tl.full((), 0, tl.int64)
    for i in range(3, 0, stride):
        total += i
    tl.store(Y, total)
  }

theorem range_captures_entry_step :
    (FP.Structural.exec model (captured "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .int 3) := by
  simp [FP.Structural.exec,captured,ComputeKernel.surfaceBody,run,step,range,rangeInt,loop,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.initial,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,State.setReg,TileShape.allIndices,State.write,Cell.read,numeric,bop,
    store,String.reduceEq,natLt] <;> norm_num [integralMod]
#axiomsClean range_captures_entry_step

theorem earlier_rebinding_changes_step :
    (FP.Structural.exec model (reassigned "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .int 3) := by
  simp [FP.Structural.exec,reassigned,ComputeKernel.surfaceBody,run,step,range,rangeInt,loop,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    IntegerLiteralAssignmentBody.initial,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,State.setReg,TileShape.allIndices,State.write,Cell.read,numeric,bop,
    store,String.reduceEq,natLt] <;> norm_num [integralMod]
#axiomsClean earlier_rebinding_changes_step

theorem negative_constant_keeps_descending_order :
    (FP.Structural.exec model (descending "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .int 6) := by
  simp only [FP.Structural.exec,descending]
  iterate 8
    all_goals try simp [ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
      IntegerLiteralAssignmentBody.initial,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,State.setReg,TileShape.allIndices,State.write,Cell.read,numeric,bop,
      store,String.reduceEq,natLt]
    all_goals try rw [rangeInt]
  all_goals norm_num [integralMod]

#axiomsClean negative_constant_keeps_descending_order

-- This existing constructor limitation is separate from range lowering.
/-- error: tl.cast: signed integer casts are not modeled yet -/
#guard_msgs in
def unsupportedNegativeFull (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    stride = tl.full((), -1, tl.int64)
    tl.store(Y, stride)
  }

/-- error: range step cannot be zero -/
#guard_msgs in
def zeroStep (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    stride = tl.full((), 0, tl.int64)
    for i in range(0, 3, stride):
        tl.store(Y, i)
  }
end
end VeriTile.Tests.KnownIntegerRangeStepBody
