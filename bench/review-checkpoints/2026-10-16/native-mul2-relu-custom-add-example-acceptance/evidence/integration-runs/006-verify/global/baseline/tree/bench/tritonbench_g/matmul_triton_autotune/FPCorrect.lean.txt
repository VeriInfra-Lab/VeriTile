import bench.tritonbench_g.matmul_triton_autotune.Proofs.FPLayout
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FPSoftware
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Full original grouped K stream, all five explicit floating input storage profiles and fixed half output. The source dot
receives the live accumulator, and every padded lane remains present. The old
row-major conditions imply output injection; no numeric law or finite-input
restriction is needed. Native dotAcc/cast/compiler binding remains separate. -/
specification full_backend_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) (hcn : p.stride_cn=1) (hbn : BN p≤p.stride_cm) :
    ∃ final, FP.Structural.exec (backend M D) (source p st) initial=some final ∧
      final.mem=(written M D p st initial).mem ∧
      (∀ i, FPValue.active p initial i →
        final.mem p.C (FPValue.cOffset p initial i)=stored st (outputValue M D p st initial i)) ∧
      (∀ r o, (r≠p.C ∨ ∀ i, FPValue.active p initial i → o≠FPValue.cOffset p initial i) →
        final.mem r o=initial.mem r o) :=
  correctness M D p st initial (fun _ _ _ _ he => original_layout_injective p initial hcn hbn he)

/-- Complete half software-bit instance with live-accumulator ascending
fp32 FMA. Both input and output conversions and activation are concrete; all results are
independent of fallback callbacks. This explicit schedule and conversion
policy are not asserted to match an unspecified GPU lowering. -/
specification half_software_correctness (rest : Algebra Word) (p : Parameters)
    (initial : State Word) (hcn : p.stride_cn=1) (hbn : BN p≤p.stride_cm) :
    ∃ final, FP.Structural.exec (backend (software rest) sequentialDotAcc) (source p .half) initial=some final ∧
      (∀ i, FPValue.active p initial i →
        final.mem p.C (FPValue.cOffset p initial i)=stored .half (bitOutput p initial i)) ∧
      (∀ r o, (r≠p.C ∨ ∀ i, FPValue.active p initial i → o≠FPValue.cOffset p initial i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨t,he,hm,hp⟩ := correctness (software rest) sequentialDotAcc p .half initial
    (fun _ _ _ _ he => original_layout_injective p initial hcn hbn he)
  exact ⟨t,he,by simpa [Post,software_reference] using hp⟩

#axiomsClean full_backend_correctness
#axiomsClean half_software_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPCorrect
