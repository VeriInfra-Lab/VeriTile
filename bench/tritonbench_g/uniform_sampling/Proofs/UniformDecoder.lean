import bench.tritonbench_g.uniform_sampling.Kernels
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TypedRandom
import VeriTile.Meta.StatementAudit

/-! The actual dtype-selected fallback helper is connected to the canonical
word decoder used by the complete source. Philox4x32 returns only uint32; the
copied uint64 branch is not a runtime alternative at these call sites. -/
namespace VeriTile.Bench.Repairs.UniformSampling.DecoderProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

theorem scale_literal :
    FP.round .fp32 FP.Config.ieeeValue (46566127342 / 10^20 : Rat) = Random.scale := by
  decide +kernel

theorem sign_fold (word : Random.Word) :
    (if word.toInt < 0 then -word.toInt-1 else word.toInt) = (Random.foldSign word : Int) := by
  rw [BitVec.toInt_eq_toNat_cond]
  have hn := word.isLt
  unfold Random.foldSign
  split <;> split <;> split <;> norm_num at * <;> omega

/-- Source bitcast, sign fold, integer conversion and scale multiplication
retain both rounding operations of the existing fixed software decoder. -/
theorem decoder_eq (word : Random.Word) :
    uniformDecoderSource.eval32 word = Random.uniform word := by
  dsimp only [UniformJitFunction.eval32]
  rw [sign_fold,scale_literal]
  rfl

/-- The typed primitive at each actual Philox4x32 call implements the selected
copied helper. The surrounding algebra receives its exact result bits. -/
theorem eval_decoder {shape : TileShape} (words : TileIndex shape → Random.Word)
    {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (operand : Op .nat shape)
    (h : FP.Structural.evalOp M p operand s = some (fun i => (words i).toNat)) :
    FP.Structural.evalOp M p (.uniform32 operand) s =
      some (fun i => M.fp32Bits ⟨(uniformDecoderSource.eval32 (words i)).bits⟩) := by
  simp only [decoder_eq]
  exact TypedRandom.eval_uniform words M p s operand h

#axiomsClean scale_literal
#axiomsClean sign_fold
#axiomsClean decoder_eq
#axiomsClean eval_decoder
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.DecoderProof
