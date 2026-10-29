import bench.tests.IntegerLiteralAssignmentBody

namespace VeriTile.Tests.LocalNatConstexprBody
open VeriTile.Triton FP.Structural
open IntegerLiteralAssignmentBody (model initial)
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Y",.i64)]
  constexpr := ["BLOCK_SIZE_IN"]

def source (Y : RegionName) (BLOCK_SIZE_IN : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2
    offsets = tl.arange(0, BLOCK_SIZE_OUT)
    tl.store(Y + offsets, offsets.to(tl.int64))
  }

/-- The declaration is evaluated with the symbolic caller dimension; it
creates the six-lane tile below without a runtime dimension register. -/
theorem six_lane_output :
    (FP.Structural.exec model (source "y" 3) initial).map
      (fun s => (s.mem "y" 5, s.mem "y" 6, s.regs .nat [] "BLOCK_SIZE_OUT")) =
        some (.mk .int 5, .mk .real 1, none) := by
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean six_lane_output
end
end VeriTile.Tests.LocalNatConstexprBody
