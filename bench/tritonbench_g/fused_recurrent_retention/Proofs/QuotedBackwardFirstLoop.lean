import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardFirstStep

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def backwardFirstTail [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) : State (Staged.Node α) :=
  if t<s.T then backwardFirstTail a s b plans initial seed pid (t+1)
    (backwardFirstBodyResult a s b plans initial u seed pid t) else u
termination_by s.T-t
decreasing_by omega

theorem backward_first_range_values [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) (ht : t ≤ s.T) (hBV : 0<s.BV)
    (covered : plans.dq.Covers s.BV) (h : BackwardFirstInvariant a s b initial seed pid t u) :
    range (Staged.algebra a plans) "i" t s.T 1
        (Program.backwardFirst (layout s b) backwardRegions seed final).body u =
      some (backwardFirstTail a s b plans initial seed pid t u) ∧
    BackwardFirstInvariant a s b initial seed pid s.T (backwardFirstTail a s b plans initial seed pid t u) := by
  by_cases active : t<s.T
  · have he := backward_first_values a s b plans seed final initial u pid t hBV covered h
    have hi := backwardFirstBodyResult_invariant a s b plans initial u seed pid t h
    obtain ⟨hrun,hinv⟩ := backward_first_range_values a s b plans initial seed final pid (t+1)
      (backwardFirstBodyResult a s b plans initial u seed pid t) (by omega) hBV covered hi
    rw [backwardFirstTail,if_pos active]
    refine ⟨?_,hinv⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active,he,bind,Option.bind_some]
    exact hrun
  · have heq : t=s.T := by omega
    rw [backwardFirstTail,if_neg active]
    refine ⟨?_,heq ▸ h⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active]
termination_by s.T-t
decreasing_by omega

#axiomsClean backward_first_range_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
