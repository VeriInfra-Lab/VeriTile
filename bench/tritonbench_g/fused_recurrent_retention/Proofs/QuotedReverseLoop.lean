import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReverseStep

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def reverseTail [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (final : Bool) (pid : ProgramId)
    (r : Nat) (u : State (Staged.Node α)) : State (Staged.Node α) :=
  if r<s.T then reverseTail a s b plans initial final pid (r+1)
    (reverseBodyResult a s b plans initial u final pid r) else u
termination_by s.T-r
decreasing_by omega

theorem reverse_range_values [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (plans : Plans) (initial : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId)
    (r : Nat) (u : State (Staged.Node α)) (hr : r ≤ s.T) (hBK : 0<s.BK) (hBV : 0<s.BV)
    (coveredK : plans.dk.Covers s.BV) (coveredV : plans.dv.Covers s.BK)
    (h : BackwardSecondInvariant a s b initial final pid r u) :
    range (Staged.algebra a plans) "_t" r s.T 1
        (Program.backwardSecond (layout s b) backwardRegions seed final).body u =
      some (reverseTail a s b plans initial final pid r u) ∧
    BackwardSecondInvariant a s b initial final pid s.T
      (reverseTail a s b plans initial final pid r u) := by
  by_cases active : r<s.T
  · have he := reverse_body_values a s b plans seed final initial u pid r active hBK hBV coveredK coveredV h
    have hi := reverseBodyResult_invariant a s b plans initial u final pid r h
    obtain ⟨hrun,hinv⟩ := reverse_range_values a s b plans initial seed final pid (r+1)
      (reverseBodyResult a s b plans initial u final pid r) (by omega) hBK hBV coveredK coveredV hi
    rw [reverseTail,if_pos active]
    refine ⟨?_,hinv⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active,he,bind,Option.bind_some]
    exact hrun
  · have heq : r=s.T := by omega
    rw [reverseTail,if_neg active]
    refine ⟨?_,heq ▸ h⟩
    rw [range]
    simp only [Nat.one_ne_zero,↓reduceIte,active]
termination_by s.T-r
decreasing_by omega

#axiomsClean reverse_range_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
