import bench.tritonbench_g.token_softmax_bloom.Proofs.IOSchedule
import bench.tritonbench_g.token_softmax_bloom.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 2000000
variable {α β : Type} [Inhabited β]

/-- Assemble all active logical output events and the exact exterior frame
from the established complete source schedule. The scalar equation is proved
independently for Real and for the selected native instruction graph. -/
theorem result (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (positive : 0 < p.width)
    (sep : Grid.Separate p) (work : List Grid.PID)
    (f : Nat → (Fin p.width → α) → Fin p.width → α)
    (vals : (io M encode p observe work).Inputs) (s : State β)
    (binding : (io M encode p observe work).InputBinding vals s)
    (unique : SoftmaxIO.Unique p work vals)
    (scalar : ∀ r (i : Fin p.width), i.val < SoftmaxIO.length p work vals r →
      observe i (outputValues M .fp32 p (Grid.state s.mem (work.get r)) (i,())) =
        f (SoftmaxIO.length p work vals r) (fun c => vals (2 : Fin 3) (r,c,())) i) :
    ∃ final, runSchedule ((io M encode p observe work).schedule vals) s = some final ∧
      (io M encode p observe work).OutputBinding vals (target M encode p observe work f vals) final ∧
      (io M encode p observe work).Frame vals s final := by
  have hu := IOBinding.unique M encode p observe work vals s binding unique
  obtain ⟨raw, rawRun, _, values, frame⟩ := Grid.whole M p positive work s.mem sep hu
  obtain ⟨final, executed, graph⟩ := IOSchedule.complete M p positive work s
  have memory : final.mem = raw := graph.trans
    (Option.some.inj ((Grid.complete M p positive work s.mem).symm.trans rawRun))
  refine ⟨final, executed, ?_, ?_⟩
  · intro port j active
    have eq : port = (0 : Fin 1) := Fin.eq_zero port
    subst port
    change mapCell (observe j.2.1) (final.mem p.output (SoftmaxIO.outputOffset p work vals j.1 j.2.1.val)) =
      Cell.mk .fp32 (f (SoftmaxIO.length p work vals j.1) (fun c => vals (2 : Fin 3) (j.1,c,())) j.2.1)
    rw [memory, ← IOBinding.offset_read M encode p observe work vals s binding j.1 (j.2.1,())]
    have live : (work.get j.1,(j.2.1,())) ∈ Grid.events p s.mem work := by
      apply (Grid.mem_events p s.mem work _).mpr
      refine ⟨List.get_mem _ _, ?_⟩
      simpa only [Grid.active, IOBinding.length_read M encode p observe work vals s binding j.1] using active
    rw [values _ live]
    change Cell.mk .fp32 (observe j.2.1 (outputValues M .fp32 p (Grid.state s.mem (work.get j.1)) (j.2.1,()))) = _
    exact congrArg (Cell.mk .fp32) (scalar j.1 j.2.1 active)
  · intro region offset outside
    rw [memory]
    apply frame
    by_cases same : region = p.output
    · right
      intro e live
      have active := (Grid.mem_events p s.mem work e).mp live
      obtain ⟨r,hr⟩ := List.mem_iff_get.mp active.1
      have ai : e.2.1.val < SoftmaxIO.length p work vals r := by
        have h := active.2
        rw [←hr] at h
        simpa only [Grid.active, IOBinding.length_read M encode p observe work vals s binding r] using h
      have h := outside.1 (0 : Fin 1) (r,e.2.1,()) ai
      have different := h.resolve_left (fun different => different same)
      rw [←hr, IOBinding.offset_read M encode p observe work vals s binding r e.2]
      exact different
    · exact Or.inl same

#axiomsClean result
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOResults
