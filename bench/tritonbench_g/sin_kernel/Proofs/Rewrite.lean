import bench.tritonbench_g.sin_kernel.Kernels
import bench.tritonbench_g.sin_kernel.Proofs.Mathematical_FP
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinKernel.Rewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000

def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    {t : State α // FP.Structural.exec (engine M) (originalKernel n) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [originalKernel,kernel,FP.Structural.exec,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLt,natLt,bop,store,Region.cast]
  rfl

/-- Exact Nat multiplication commutation; opaque FP32 sine and arbitrary
inactive load values are identical in the two complete executions. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    FP.Structural.exec (engine M) (originalKernel n) s =
      FP.Structural.exec (engine M) (candidateKernel n) s := by
  change run (engine M)
      (VeriTile.Bench.TritonBenchG.SinKernel.FPProof.sourcePrefix Bool.false blockSize ++
        (originalKernel n).surfaceBody.drop 2) s =
    run (engine M)
      (VeriTile.Bench.TritonBenchG.SinKernel.FPProof.sourcePrefix Bool.true blockSize ++
        (originalKernel n).surfaceBody.drop 2) s
  exact VeriTile.Bench.TritonBenchG.SinKernel.FPProof.prefix_run M s blockSize _

#axiomsClean originalExecution
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.SinKernel.Rewrite
