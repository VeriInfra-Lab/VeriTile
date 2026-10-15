import bench.tritonbench_g.token_attn_llama2.Proofs.FPContract
import bench.tritonbench_g.token_attn_llama2.Proofs.FPSumProfile
namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

/-- Complete arbitrary-scale stage with concrete scalar arithmetic/tree and a
named scalar-encoding parameter. No other fallback callback affects output. -/
theorem software_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (encode : ℝ → Word) (p : Parameters)
    (s : State Word) (hw : WritesApart p s) :
    ∃ final, FP.Structural.exec (softwareBackend rest cfg plans encode) (source p) s = some final ∧
      Post p s final (wordScore cfg plans (scaleWord encode p.scale) p s) := by
  obtain ⟨final, he, hp⟩ := correctness (softwareBackend rest cfg plans encode) p s hw
  refine ⟨final, he, ?_⟩
  simpa only [Post, score_software] using hp

/-- An executable scalar binding for every exact finite host rational.
The full backend contract above still covers arbitrary Real parameters. This
specialization does not claim a native host-ABI conversion or GPU reduction tree. -/
theorem rational_correctness [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (q : Rat) (p : Parameters)
    (s : State Word) (hw : WritesApart p s) :
    ∃ final, FP.Structural.exec (softwareBackend rest cfg plans (rationalEncoder cfg q))
      (source {p with scale := (q : ℝ)}) s = some final ∧
      Post {p with scale := (q : ℝ)} s final
        (wordScore cfg plans (FP.round .fp32 cfg q) {p with scale := (q : ℝ)} s) := by
  have hwu : WritesApart {p with scale := (q : ℝ)} s := hw
  obtain ⟨final, he, hp⟩ := software_correctness rest cfg plans (rationalEncoder cfg q)
    {p with scale := (q : ℝ)} s hwu
  refine ⟨final, he, ?_⟩
  simpa only [rational_scale] using hp

/-- Inactive blocks preserve every tagged memory cell, independently of all
numerical callbacks. The original source loop performs no score iteration. -/
theorem inactive_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (hb : ¬ blockActive p s) :
    ∃ final, FP.Structural.exec M (source p) s = some final ∧ final.mem = s.mem := by
  refine ⟨(execution M p s).val, (execution M p s).property, ?_⟩
  rw [execution_memory]
  simp [written, hb]

#axiomsClean software_correctness
#axiomsClean rational_correctness
#axiomsClean inactive_memory
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
