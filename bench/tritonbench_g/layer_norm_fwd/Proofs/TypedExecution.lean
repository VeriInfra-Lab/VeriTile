import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedAffine

/-! Concrete successful complete-source typed executions in an explicit
consistency algebra. Both normalization modes execute every optional feature,
from empty registers at PID 1, with a two-column row in a four-lane block.
The proof composes actual source phases. This consistency witness is not an
IEEE/native arithmetic model; universal real correctness and whole-state typed
FP equivalence are separate theorems. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- All source statements execute, starting from the initial memory and empty
registers. Phase invariants are conclusions of preceding executions. -/
theorem source_run (rms : Bool) (p : ℝ) :
    ∃ t, run (engine model)
      (original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody initial = some t := by
  obtain ⟨s₁, h₁, hr₁⟩ := prefix_success rms p
  obtain ⟨s₂, h₂, hr₂⟩ := firstStream_success rms p s₁ hr₁
  obtain ⟨s₃, h₃, hr₃⟩ := secondStream_success rms p s₂ hr₂
  obtain ⟨s₄, h₄, hr₄⟩ := normalization_success rms p s₃ hr₃
  obtain ⟨s₅, h₅⟩ := affine_success rms p s₄ hr₄
  let code := (original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody
  change run (engine model) (code.take 10) initial = some s₁ at h₁
  change run (engine model) ((code.drop 10).take 2) s₁ = some s₂ at h₂
  change run (engine model) ((code.drop 12).take 3) s₂ = some s₃ at h₃
  change run (engine model) ((code.drop 15).take 3) s₃ = some s₄ at h₄
  change run (engine model) (code.drop 18) s₄ = some s₅ at h₅
  refine ⟨s₅, ?_⟩
  change run (engine model) code initial = some s₅
  rw [← List.take_append_drop 10 code, run_append, h₁, Option.bind_some]
  rw [← List.take_append_drop 2 (code.drop 10), run_append, h₂, Option.bind_some]
  simp only [List.drop_drop, Nat.reduceAdd]
  rw [← List.take_append_drop 3 (code.drop 12), run_append, h₃, Option.bind_some]
  simp only [List.drop_drop, Nat.reduceAdd]
  rw [← List.take_append_drop 3 (code.drop 15), run_append, h₄, Option.bind_some]
  simpa only [List.drop_drop, Nat.reduceAdd] using h₅

theorem nonempty_pair (rms : Bool) (p : ℝ) :
    ∃ final,
      FP.Structural.exec (engine model)
        (original (flags rms) regions strides 2 2 1 p 4 legal_block) initial = some final ∧
      FP.Structural.exec (engine model)
        (candidate (flags rms) regions strides 2 2 1 p 4 legal_block) initial = some final := by
  obtain ⟨final, h⟩ := source_run rms p
  exact ⟨final, h, (FPProof.whole_state (flags rms) regions strides 2 2 1 p 4 legal_block
    (engine model) initial).symm.trans h⟩

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
