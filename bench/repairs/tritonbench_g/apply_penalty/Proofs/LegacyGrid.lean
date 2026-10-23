/- Optional historical whole-grid/native transport results. These are not
required public kernel acceptance and primitive conformance is not a deferred
numerical experiment. -/
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Real

namespace VeriTile.Bench.Repairs.ApplyPenalty.LegacyGrid
open VeriTile.Triton
noncomputable section

theorem python_checks_iff (a : Arguments) : checkInputs a=.ok () ↔ InputContract a :=
  checkInputs_iff a

/-- Complete mathematical source launch in any whole-program order. Selected
coordinates use entry operands; all unselected logits and other raw cells are
preserved. `binding` only identifies input reads with tensor observations. -/
theorem apply_penalty_correct (a : Arguments) (h : InputContract a) (input : Inputs ℝ)
    (s : BlockState) (binding : RealProof.ReadOnlyBinding a input s)
    (logits : input.logits=s.readMem "logits") (order : List Nat)
    (schedule : order.Perm (List.range a.batches)) :
    ∃ t,RealProof.runSource a order s=some t ∧
      (∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
        t.readMem "logits" (b*a.vocab+a.token k)=
          penalty (input.logits (b*a.vocab+a.token k)) (a.counts k).toInt
            (input.presence b) (input.frequency b) (input.repetition b)) ∧
      (∀ q,(∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
        q≠b*a.vocab+a.token k) → t.readMem "logits" q=input.logits q) ∧
      (∀ r o,r≠"logits" → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hv,hf⟩ := RealProof.whole_source a h input order
    (fun b hb => List.mem_range.mp (schedule.mem_iff.mp hb)) s binding
  have result := complete_grid RealProof.ops a h input order schedule
  refine ⟨t,he,?_,?_,hf⟩
  · intro b hb k hk he
    rw [hv,← logits]
    exact (result.1 b hb k hk he).trans (RealProof.arithmetic_formula _ _ _ _ _)
  · intro q miss
    rw [hv,← logits]
    exact result.2 q miss

theorem wrapper_formula (a : Arguments) (h : InputContract a) (input : Inputs ℝ) :
    wrapperRun RealProof.ops a input=.ok (gridRun RealProof.ops a input (List.range a.batches)) ∧
    (∀ b,b<a.batches → ∀ k,a.start b≤k → k<a.stop b →
      gridRun RealProof.ops a input (List.range a.batches) (b*a.vocab+a.token k)=
        penalty (input.logits (b*a.vocab+a.token k)) (a.counts k).toInt
          (input.presence b) (input.frequency b) (input.repetition b)) := by
  constructor
  · simp [wrapperRun,(checkInputs_iff a).mpr h]
  · intro b hb k hk he
    exact ((complete_grid RealProof.ops a h input _ (List.Perm.refl _)).1 b hb k hk he).trans
      (RealProof.arithmetic_formula _ _ _ _ _)

#axiomsClean python_checks_iff
#axiomsClean apply_penalty_correct
#axiomsClean wrapper_formula
end
end VeriTile.Bench.Repairs.ApplyPenalty.LegacyGrid
