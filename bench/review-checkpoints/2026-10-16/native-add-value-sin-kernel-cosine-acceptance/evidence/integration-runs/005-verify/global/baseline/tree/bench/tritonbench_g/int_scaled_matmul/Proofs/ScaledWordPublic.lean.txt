import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordShape
import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordWitness
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
local instance scaledPublicDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Complete accepted kernel2: both EVEN_K branches and dynamic K, signed8
inputs, modular signed32 dot/update, rounded integer-to-fp32 and fp32 scale
multiplication, followed by defined toward-zero signed32 output conversion.
Only active lanes require a defined conversion. Source masks, uninitialized
inactive scale values and all original indexing are retained. This is a
software bit-model contract; compiler/GPU refinement is separate. -/
theorem scaled_kernel_machine_correctness
    (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word)
    (hcover : p.inner ≤ p.steps*p.blockInner)
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (hd : InputDefined cfg p initial) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial=some final ∧
      Post p initial final (expected cfg p initial) ∧
      (∀ i, active p initial i → ∃ q, (scaled cfg p initial i).toRat?=some q ∧
        TowardZero q (expected cfg p initial i) ∧ SignedRange 32 (expected cfg p initial i)) ∧
      (∀ j : Fin p.inner, ∃ k, ∃ lane : Fin p.blockInner, k<p.steps ∧
        (p.even=Bool.true ∨ inputInner p initial lane < p.inner-k*p.blockInner) ∧
        k*p.blockInner+inputInner p initial lane=j.val) ∧
      (p.even=Bool.true → ∀ k, k<p.steps → ∀ lane : Fin p.blockInner,
        k*p.blockInner+lane.val<p.inner) := by
  obtain ⟨final,he,hp⟩ := correctness rest cfg u p initial hd (output_injective p initial)
  exact ⟨final,he,hp,expectedSpec cfg p initial hd,inner_coverage p initial hcover,
    fun hb k hk lane => even_lane_in_bounds p heven hb k hk lane⟩
#axiomsClean scaled_kernel_machine_correctness
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
