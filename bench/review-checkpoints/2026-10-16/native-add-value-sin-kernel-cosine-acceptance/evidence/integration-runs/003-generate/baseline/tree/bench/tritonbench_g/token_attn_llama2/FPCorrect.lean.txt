import bench.tritonbench_g.token_attn_llama2.Proofs.FPShape
import bench.tritonbench_g.token_attn_llama2.Proofs.FPWitness

namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section

/-! Complete Q/K score stage, including the original dynamic 0/1 loop,
metadata gather, all dimensions/strides, and masked readback/frame.

The arbitrary-Real scale theorem is backend-relative. The software theorem
names its scalar encoder explicitly. The rational specialization closes that
encoder using direct fp32 rounding; it does not assert a native ABI or GPU
reduction schedule. The ordinary bit witness is a regression, not a scope
restriction or an upgrade of the arbitrary-Real theorem. -/

specification token_attn_llama2_fp_backend_correctness
    {α : Type} [Inhabited α] (M : Algebra α) (p : FPValue.Parameters)
    (s : State α) (hw : FPValue.WritesApart p s) :
    ∃ final, FP.Structural.exec M (FPValue.source p) s = some final ∧
      FPValue.Post p s final (FPValue.score M p s) :=
  FPValue.correctness M p s hw

specification token_attn_llama2_fp_software_correctness
    [Inhabited FPValue.Word] (rest : Algebra FPValue.Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (encode : ℝ → FPValue.Word)
    (p : FPValue.Parameters) (s : State FPValue.Word) (hw : FPValue.WritesApart p s) :
    ∃ final, FP.Structural.exec (FPValue.softwareBackend rest cfg plans encode)
        (FPValue.source p) s = some final ∧
      FPValue.Post p s final
        (FPValue.wordScore cfg plans (FPValue.scaleWord encode p.scale) p s) :=
  FPValue.software_correctness rest cfg plans encode p s hw

specification token_attn_llama2_fp_rational_correctness
    [Inhabited FPValue.Word] (rest : Algebra FPValue.Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (q : Rat) (p : FPValue.Parameters)
    (s : State FPValue.Word) (hw : FPValue.WritesApart p s) :
    ∃ final, FP.Structural.exec
        (FPValue.softwareBackend rest cfg plans (FPValue.rationalEncoder cfg q))
        (FPValue.source {p with scale := (q : ℝ)}) s = some final ∧
      FPValue.Post {p with scale := (q : ℝ)} s final
        (FPValue.wordScore cfg plans (FP.round .fp32 cfg q) {p with scale := (q : ℝ)} s) :=
  FPValue.rational_correctness rest cfg plans q p s hw

theorem token_attn_llama2_fp_source_projection (p : FPValue.Parameters) :
    (FPValue.source p).eraseDType.surfaceBody = (FPValue.legacySource p).eraseDType.surfaceBody :=
  FPValue.source_projection p

theorem token_attn_llama2_fp_inactive_memory
    {α : Type} [Inhabited α] (M : Algebra α) (p : FPValue.Parameters)
    (s : State α) (hb : ¬ FPValue.blockActive p s) :
    ∃ final, FP.Structural.exec M (FPValue.source p) s = some final ∧ final.mem = s.mem :=
  FPValue.inactive_memory M p s hb

theorem token_attn_llama2_fp_bit_success
    [Inhabited FPValue.Word] (rest : Algebra FPValue.Word) :
    ∃ final, FP.Structural.exec
        (FPValue.softwareBackend rest .ieeeValue ReductionPlan.withSeed
          (FPValue.rationalEncoder .ieeeValue FPValue.witnessScale))
        (FPValue.source FPValue.witnessParameters) FPValue.witnessInitial = some final ∧
      final.mem "Out" 5 = .mk .fp32 FPValue.one ∧
      final.mem "Out" 6 = FPValue.witnessInitial.mem "Out" 6 ∧
      final.mem "Q" 0 = FPValue.witnessInitial.mem "Q" 0 ∧
      FPValue.witnessScale * FPValue.witnessScale ≠ 1 :=
  FPValue.nonempty_bit_success rest

#axiomsClean token_attn_llama2_fp_backend_correctness
#axiomsClean token_attn_llama2_fp_software_correctness
#axiomsClean token_attn_llama2_fp_rational_correctness
#axiomsClean token_attn_llama2_fp_source_projection
#axiomsClean token_attn_llama2_fp_inactive_memory
#axiomsClean token_attn_llama2_fp_bit_success
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2
