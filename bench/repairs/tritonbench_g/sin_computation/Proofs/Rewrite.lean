import bench.repairs.tritonbench_g.sin_computation.Kernels
import bench.tritonbench_g.sin_computation.Proofs.FP
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000

def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (X O : RegionName) (n : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (kernel X O n blockSize) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [kernel,FP.Structural.exec,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLt,natLt,bop,store,Region.cast]
  rfl

/-- Exact Nat multiplication commutation; opaque FP32 sine and arbitrary
inactive load values are identical in the two complete executions. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (X O : RegionName) (n : Nat) (s : State α) :
    FP.Structural.exec (engine M) (kernel X O n blockSize) s =
      FP.Structural.exec (engine M) (indexCommutedKernel X O n blockSize) s := by
  change run (engine M)
      (VeriTile.Bench.TritonBenchG.SinComputation.FPProof.sourcePrefix Bool.false blockSize ++
        (kernel X O n blockSize).surfaceBody.drop 2) s =
    run (engine M)
      (VeriTile.Bench.TritonBenchG.SinComputation.FPProof.sourcePrefix Bool.true blockSize ++
        (kernel X O n blockSize).surfaceBody.drop 2) s
  exact VeriTile.Bench.TritonBenchG.SinComputation.FPProof.prefix_run M s blockSize _

#axiomsClean originalExecution
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Rewrite
