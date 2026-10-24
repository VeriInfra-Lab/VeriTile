import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordPublic
import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordPublic

namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.FPCorrect
open VeriTile.Triton FP.Structural
noncomputable section

/-- The original block-pointer integer kernel, including unspecified boundary
lanes, concrete signed-byte input and 32-bit wrapping accumulation. -/
specification block_pointer_word_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : BlockWord.Parameters) (s : State α)
    (hBK : 0 < p.blockInner) (hBytes : BlockWord.ByteInputs p s)
    (hw : ∀ i j, BlockWord.active p s i → BlockWord.active p s j →
      BlockWord.outputOffset p s i = BlockWord.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (BlockWord.backend M) (BlockWord.source p) s = some final ∧
      BlockWord.Post p s final (BlockWord.outputValue p s) ∧
      (∀ k, k < BlockWord.steps p → ∀ i : BlockWord.AIndex p,
        BlockWord.InputReadback s (BlockWord.aPointer p s k) i) ∧
      (∀ k, k < BlockWord.steps p → ∀ i : BlockWord.BIndex p,
        BlockWord.InputReadback s (BlockWord.bPointer p s k) i) := by
  exact BlockWord.machine_correctness M p s hBK hBytes hw

section Scaled
open ScaledWord FP.IntegerConversion
local instance scaledPublicDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Complete dynamic scaled integer kernel with both EVEN_K branches.
The final conversion is actual toward-zero int32, defined on active lanes. -/
specification scaled_word_correctness
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
  exact ScaledWord.scaled_kernel_machine_correctness rest cfg u p initial hcover heven hd
#axiomsClean scaled_word_correctness
end Scaled

#axiomsClean block_pointer_word_correctness
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.FPCorrect
