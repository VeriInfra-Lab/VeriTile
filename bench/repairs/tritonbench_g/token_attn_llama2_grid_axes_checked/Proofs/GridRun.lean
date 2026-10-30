import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceContract
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Geometry

/-! Sequential whole-grid execution, with independent initial-memory score
readback. No successful run, final memory, or numerical equality is assumed. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.NativeGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

open Whole
variable {α : Type} [Inhabited α]


private theorem references_eq (M : Algebra α) (p : Parameters) (sep : Separated p)
    (s t : State α) (hp : s.pids = t.pids) (hr : ReadsEq p s t) :
    active p s = active p t ∧ outputOffset p s = outputOffset p t ∧
      score M p s = score M p t := by
  have hq := hr p.q sep.query
  have hk := hr p.k sep.key
  have hl := hr p.locations.cast sep.locations
  have hs := hr p.starts.cast sep.starts
  have hn := hr p.lengths.cast sep.lengths
  have prod_eq : products M p s = products M p t := by
    funext i
    simp only [products, queryValue, keyValue, queryOffset, keyOffset, keyLocation,
      active, windowIndex, length, token, hp, hq, hk, hl, hn]
  refine ⟨?_, ?_, ?_⟩
  all_goals funext i
  all_goals simp only [active, windowIndex, length, token, outputOffset, score,
    products, queryValue, keyValue, queryOffset, keyOffset, keyLocation,
    hp, hq, hk, hl, hs, hn, prod_eq]


private theorem start_reads (p : Parameters) (g : Whole.Grid) (i : Program)
    (s t : State α) (h : ReadsEq p s t) : ReadsEq p (start g i s) (start g i t) := h

def Post (M : Program → Algebra α) (p : Parameters) (g : Whole.Grid) (jobs : List Program)
    (s t : State α) : Prop :=
  (∀ a ∈ jobs, ∀ i, active p (start g a s) i →
    t.mem p.output (outputOffset p (start g a s) i) = .mk .fp32 (score (M a) p (start g a s) i)) ∧
  (∀ r o, Outside p g jobs s r o → t.mem r o = s.mem r o)

def runPrograms (M : Program → Algebra α) (p : Parameters) (g : Whole.Grid) :
    List Program → State α → Option (State α)
  | [], s => some s
  | a :: jobs, s => (FP.Structural.exec (sourceBackend (M a)) (source p) (start g a s)).bind
      (runPrograms M p g jobs)

/-- Every execution uses the previous program's memory, not an isolated copy. -/
theorem run_correct (M : Program → Algebra α) (p : Parameters) (sep : Separated p)
    (g : Whole.Grid) (jobs : List Program) (s : State α) (unique : Unique p g jobs s) :
    ∃ t, runPrograms M p g jobs s = some t ∧ Post M p g jobs s t := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,(by simp [Post,Outside])⟩
  | cons a jobs ih =>
    have hw : WritesApart p (start g a s) := by
      intro i j hi hj he
      exact (unique a (by simp) a (by simp) i j hi hj he).2
    obtain ⟨next, he, hv, hf⟩ := VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.SourceCorrectness.correctness (M a) p (start g a s) hw
    have hr : ReadsEq p next s := by
      intro r hn
      funext o
      exact hf r o (Or.inr (Or.inl hn))
    have refs (b : Program) := references_eq (M b) p sep (start g b next) (start g b s)
      rfl (start_reads p g b next s hr)
    have uniqNext : Unique p g jobs next := by
      intro b hb c hc i j hi hj hij
      rw [(refs b).1] at hi
      rw [(refs c).1] at hj
      rw [(refs b).2.1, (refs c).2.1] at hij
      exact unique b (by simp [hb]) c (by simp [hc]) i j hi hj hij
    obtain ⟨t, ht, htv, htf⟩ := ih next uniqNext
    refine ⟨t, ?_, ?_, ?_⟩
    · simp only [runPrograms,he,Option.bind_some]
      exact ht
    · intro b hb i hi
      by_cases hbJobs : b ∈ jobs
      · have h := htv b hbJobs i (by rw [(refs b).1]; exact hi)
        simpa only [(refs b).2.1, (refs b).2.2] using h
      · have hab : b = a := by simpa [hbJobs] using hb
        subst b
        have outside : Outside p g jobs next p.output (outputOffset p (start g a s) i) := by
          right
          intro b hb j hj heq
          rw [(refs b).1] at hj
          rw [(refs b).2.1] at heq
          have same := (unique a (by simp) b (by simp [hb]) i j hi hj heq).1
          exact hbJobs (same ▸ hb)
        rw [htf _ _ outside]
        exact hv i (active_block p _ i hi) hi
    · intro r o ho
      have outTail : Outside p g jobs next r o := by
        rcases ho with ho | ho
        · exact Or.inl ho
        · right
          intro b hb i hi
          rw [(refs b).1] at hi
          rw [(refs b).2.1]
          exact ho b (by simp [hb]) i hi
      rw [htf r o outTail]
      apply hf
      rcases ho with ho | ho
      · exact Or.inr (Or.inl ho)
      · exact Or.inr (Or.inr (fun i hi => ho a (by simp) i hi))

#axiomsClean run_correct
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.NativeGrid
