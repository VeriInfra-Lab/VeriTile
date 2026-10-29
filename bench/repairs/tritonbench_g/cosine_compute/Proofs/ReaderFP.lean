import bench.repairs.tritonbench_g.cosine_compute.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.CosineCompute.ReaderProof
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

theorem index_commuted_equiv (n : Nat) :
    (⟨originalKernel n, []⟩ : Program) ≡[[]] (⟨candidateKernel n, []⟩ : Program) :=by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  let result := Rewrite.originalExecution M n s
  exact ⟨result.val,result.val,result.property,
    (Rewrite.execution_eq M n s).symm.trans result.property,rfl⟩

end
end VeriTile.Bench.Repairs.CosineCompute.ReaderProof
