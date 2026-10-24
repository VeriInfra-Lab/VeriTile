import bench.tritonbench_g.matmul_kernel.Proofs.FPLayout
import bench.tritonbench_g.matmul_kernel.Proofs.FPSoftware
import bench.tritonbench_g.matmul_kernel.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Full original 2-D program / arbitrary K stream. Every output lane stores,
all source loads are unmasked, and three-argument dot receives live carry.
Only the original BN≤4096 layout requirement is used. Numerical primitives
have no algebraic laws; their native implementation correspondence is separate. -/
specification half_backend_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) (hbn : BN p≤4096) :
    ∃ final, FP.Structural.exec (backend M D) (source p .half) initial=some final ∧
      final.mem=(written M D p .half initial).mem ∧
      (∀ i, final.mem p.C (FPValue.cOffset p initial i)=stored .half (outputValue M D p .half initial i)) ∧
      (∀ r o, (r≠p.C ∨ ∀ i, o≠FPValue.cOffset p initial i) → final.mem r o=initial.mem r o) := by
  obtain ⟨t,he,hm,hp⟩ := correctness M D p .half initial
    (fun _ _ _ _ he => original_layout_injective p initial hbn he)
  exact ⟨t,he,hm,by simpa [Post,FPValue.active] using hp⟩
/-- Complete binary16-storage / binary32 live-FMA software-bit contract.
The explicit ascending FMA schedule and IEEE conversion profile are closed,
including NaNs, infinities, subnormals and signed zeros; no identity-rounding
or finite-input hypothesis is imposed. This is not a GPU schedule theorem. -/
specification half_software_correctness (rest : Algebra Word) (p : Parameters)
    (initial : State Word) (hbn : BN p≤4096) :
    ∃ final, FP.Structural.exec (backend (software rest) sequentialDotAcc) (source p .half) initial=some final ∧
      (∀ i, final.mem p.C (FPValue.cOffset p initial i)=stored .half (bitOutput p initial i)) ∧
      (∀ r o, (r≠p.C ∨ ∀ i, o≠FPValue.cOffset p initial i) → final.mem r o=initial.mem r o) := by
  obtain ⟨t,he,hm,hp⟩ := correctness (software rest) sequentialDotAcc p .half initial
    (fun _ _ _ _ he => original_layout_injective p initial hbn he)
  exact ⟨t,he,by simpa [Post,FPValue.active,software_reference] using hp⟩
#axiomsClean half_backend_correctness
#axiomsClean half_software_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPCorrect
