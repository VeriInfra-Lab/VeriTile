import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardStep

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def forwardTail [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) : State (Staged.Node α) :=
  if t<s.T then forwardTail a s b plans initial seed pid (t+1)
    (forwardBodyResult a s b plans initial u seed pid t) else u
termination_by s.T-t
decreasing_by omega

theorem forward_range_values [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (t : Nat) (u : State (Staged.Node α)) (ht : t ≤ s.T) (hBK : 0<s.BK)
    (covered : plans.output.Covers s.BK) (h : ForwardInvariant a s b initial seed pid t u) :
    range (Staged.algebra a plans) "_t" t s.T 1
        (Program.forwardParts (layout s b) forwardRegions seed final).body u =
      some (forwardTail a s b plans initial seed pid t u) ∧
    ForwardInvariant a s b initial seed pid s.T (forwardTail a s b plans initial seed pid t u) := by
  by_cases active : t<s.T
  · have he := forward_body_values a s b plans seed final initial u pid t hBK covered h
    have hi := forwardBodyResult_invariant a s b plans initial u seed pid t h
    obtain ⟨hrun,hinv⟩ := forward_range_values a s b plans initial seed final pid (t+1)
      (forwardBodyResult a s b plans initial u seed pid t) (by omega) hBK covered hi
    rw [forwardTail,if_pos active]
    refine ⟨?_,hinv⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active,he,bind,Option.bind_some]
    exact hrun
  · have heq : t=s.T := by omega
    rw [forwardTail,if_neg active]
    refine ⟨?_,heq ▸ h⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active]
termination_by s.T-t
decreasing_by omega

#axiomsClean forward_range_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
