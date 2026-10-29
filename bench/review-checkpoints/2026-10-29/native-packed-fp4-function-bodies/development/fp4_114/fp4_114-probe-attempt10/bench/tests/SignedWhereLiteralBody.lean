import bench.tests.IntegerLiteralAssignmentBody
namespace VeriTile.Tests.SignedWhereLiteralBody
open VeriTile.Triton FP.Structural
open IntegerLiteralAssignmentBody (model)
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.i32),("Y",.i32)]
def source (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    selected = tl.where(x == 0, 0, x)
    sign = tl.full([], 8, tl.int32)
    result = selected | sign
    tl.store(Y, result)
  }
def initial : State Int where
  mem := fun _ _ => .mk .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem signed_selection :
    (FP.Structural.exec model (source "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .int 8) := by
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,numericEq,bop,store,String.reduceEq]
#axiomsClean signed_selection
end
end VeriTile.Tests.SignedWhereLiteralBody
