import bench.repairs.tritonbench_g.cosine_compute.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CosineCompute.Rewrite
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

/-- Exact index commutation from arbitrary states. The opaque cosine and FP32
conversion retain their original operands on active and inactive lanes. -/
theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α) (n : Nat) (s : State α) :
    FP.Structural.exec (engine M) (originalKernel n) s =
      FP.Structural.exec (engine M) (candidateKernel n) s := by
  change Contextual M s [] ((originalKernel n).surfaceBody.drop 1)
    (.assign .nat [blockSize] "offset" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.programId 0) (.constNat blockSize)) (.arange blockSize))))
    (.assign .nat [blockSize] "offset" (.alg (.add .nat .scalarL
      (.mul .nat .nil (.constNat blockSize) (.programId 0)) (.arange blockSize))))
  fp_prove

#axiomsClean originalExecution
#axiomsClean execution_eq
end
end VeriTile.Bench.Repairs.CosineCompute.Rewrite
