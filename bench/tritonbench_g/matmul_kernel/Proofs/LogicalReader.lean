import bench.tritonbench_g.matmul_kernel.Proofs.PublicIO
import bench.tritonbench_g.matmul_kernel.IO

namespace VeriTile.Bench.Repairs.MatmulKernel.ReaderProof
open VeriTile.Triton FP.Structural
open ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Exact equality to the previous independent target, for every raw input. -/
theorem target_eq (p : Config) : (view p).lift (fun xs r c => ∑ t : Fin p.K, xs.A r t * xs.B t c) =
    PublicIO.expected p (fun a b r n => ∑ t ∈ Finset.range p.K, a r t * b t n) := by
  funext xs o j
  change ChannelValues ℝ [⟨.real,[p.M,p.K],"A"⟩,⟨.real,[p.K,p.N],"B"⟩] at xs
  fin_cases o
  rcases j with ⟨r,c,⟨⟩⟩
  change (∑ t : Fin p.K, (ReaderIO.inputs p xs).A r t * (ReaderIO.inputs p xs).B t c) =
    ∑ t ∈ Finset.range p.K, PublicIO.arrayA p xs r.val t * PublicIO.arrayB p xs t c.val
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro t _
  simp [ReaderIO.inputs, PublicIO.arrayA, PublicIO.arrayB, r.isLt, c.isLt, t.isLt] <;> rfl

/-- The complete old/new judgments are equivalent, including source execution,
source safety, every output and the exact exterior frame. -/
theorem judgment_iff (p : Config) :
    (view p ⊨ fun xs r c => ∑ t : Fin p.K, xs.A r t * xs.B t c) ↔
      (PublicIO.io p KernelReal.scalar (KernelScalar.dotAcc KernelReal.dot)).Implements (PublicIO.expected p (fun a b r n => ∑ t ∈ Finset.range p.K, a r t * b t n)) := by
  rw [LogicalProgramIO.implements_iff, target_eq] <;> rfl

theorem correct (p : Config) (hc : p.Valid) :
    (view p ⊨ fun xs r c => ∑ t : Fin p.K, xs.A r t * xs.B t c) :=
  (judgment_iff p).mpr (PublicIO.implements p KernelReal.scalar (KernelScalar.dotAcc KernelReal.dot)
    (KernelReal.target p.K) (KernelInputs.real p hc))
end
end VeriTile.Bench.Repairs.MatmulKernel.ReaderProof
