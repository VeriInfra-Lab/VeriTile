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
    sign = (tl.full([], 8, tl.int32)).to(tl.int32)
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

def negativeBranch (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    selected = tl.where(x == 0, x, -2147483648)
    sign = (tl.full([], 1, tl.int32)).to(tl.int32)
    result = selected | sign
    tl.store(Y, result)
  }

theorem minimum_signed_branch :
    (FP.Structural.exec model (negativeBranch "x" "y")
      {initial with mem := fun _ _ => .mk .int 1}).map
      (fun s => s.mem "y" 0) = some (.mk .int (-2147483647)) := by
  simp [FP.Structural.exec,negativeBranch,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,numericEq,bop,store,String.reduceEq]
#axiomsClean minimum_signed_branch

/-- error: tl.where integer scalar exceeds the signed tensor dtype range -/
#guard_msgs in
def oversized (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    selected = tl.where(x == 0, 2147483648, x)
    tl.store(Y, selected)
  }

/-- error: tl.where integer scalar exceeds the signed tensor dtype range -/
#guard_msgs in
def undersized (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    selected = tl.where(x == 0, x, -2147483649)
    tl.store(Y, selected)
  }

end
end VeriTile.Tests.SignedWhereLiteralBody
