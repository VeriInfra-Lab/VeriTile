import bench.tritonbench_g.chunk_retention.Sources.KernelModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelFold
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)

/-- Ascending scalar recurrence. This contains no interpreter state or syntax. -/
def sequenceValue (update : Nat → β → β) (initial : β) : Nat → β
  | 0 => initial
  | n+1 => update n (sequenceValue update initial n)

theorem range_sequenceValue (update : Nat → β → β) (initial : β) (cur stop : Nat)
    (h : cur≤stop) :
    foldRange update cur stop 1 (sequenceValue update initial cur)=sequenceValue update initial stop := by
  by_cases hs : cur<stop
  · rw [foldRange]
    simp only [show (1:Nat)≠0 by decide,hs,↓reduceIte]
    exact range_sequenceValue update initial (cur+1) stop (by omega)
  · have he : cur=stop := by omega
    subst cur
    rw [foldRange]
    simp
termination_by stop-cur
decreasing_by omega

theorem from_zero (update : Nat → β → β) (initial : β) (stop : Nat) :
    foldRange update 0 stop 1 initial=sequenceValue update initial stop :=
  range_sequenceValue update initial 0 stop (Nat.zero_le stop)

theorem sequenceValue_map (f : α → β) (u : Nat → α → α) (v : Nat → β → β)
    (commutes : ∀ n a, f (u n a)=v n (f a)) (initial : α) (n : Nat) :
    f (sequenceValue u initial n)=sequenceValue v (f initial) n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sequenceValue,commutes,ih]

#axiomsClean from_zero
#axiomsClean sequenceValue_map
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelFold
