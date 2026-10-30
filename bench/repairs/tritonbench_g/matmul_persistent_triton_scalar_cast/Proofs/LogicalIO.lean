import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.IO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicResult

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.LogicalIOProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating PublicData
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem headline_iff (p : Core.Parameters) :
    (logicalIO Real.arithmetic p ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin p.problem.K, a row k * b k col) ↔
    (io Real.arithmetic p).Implements (mathematical p) := by
  have target : (logicalIO Real.arithmetic p).lift
      (fun ⟨a,b⟩ row col => ∑ k : Fin p.problem.K, a row k * b k col) = mathematical p := by
    funext xs port j
    fin_cases port
    rfl
  change (io Real.arithmetic p).Implements _ ↔ _
  rw [target]

theorem implements (p : Core.Parameters) (b : RawKernel.Buffers)
    (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS) :
    (logicalIO Real.arithmetic p ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin p.problem.K, a row k * b k col) :=
  (headline_iff p).2 (real_correct p b layout sms)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.LogicalIOProof
