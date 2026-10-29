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
    TileShape.allIndices,List.finRange_succ,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean six_lane_output

/-- Interleave and the local constexpr dimension must describe the same
symbolic tile, before any concrete block size is chosen. -/
def interleavedSource (X Y : RegionName) (BLOCK_SIZE_IN : Nat) : ComputeKernel :=
  with_triton_signature ({signature with pointers := [("X",.i64),("Y",.i64)]} : CallSignature) in triton {
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2
    offsets = tl.arange(0, BLOCK_SIZE_IN)
    x = tl.load(X + offsets)
    y = tl.interleave(x, x)
    output_offsets = tl.arange(0, BLOCK_SIZE_OUT)
    tl.store(Y + output_offsets, y)
  }

theorem interleaved_output :
    (FP.Structural.exec model (interleavedSource "x" "y" 2)
      {initial with mem := fun _ _ => .mk .int 7}).map
      (fun s => (s.mem "y" 3, s.mem "y" 4)) =
        some (.mk .int 7, .mk .int 7) := by
  simp [FP.Structural.exec,interleavedSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,List.finRange_succ,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean interleaved_output

end
end VeriTile.Tests.LocalNatConstexprBody
