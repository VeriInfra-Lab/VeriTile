import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeBits

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeControl
open NativeDomain NativeBits
set_option maxHeartbeats 3000000

def nextBits (old count : BitVec 32) :=
  if old == count-bits 1 then bits 0 else old+bits 1

/-- Both eagerly evaluated integer arms and the comparison retain their
source meaning, including old=-1 and the wrap from last K to zero. -/
theorem next_exact (p : Parameters) (h : Admissible p) (old : Int)
    (hold : -1 ≤ old ∧ old < (count p : Int))
    (oldBits countBits : BitVec 32) (ho : Exact oldBits old) (hc : Exact countBits (count p)) :
    Exact (countBits-bits 1) ((count p : Int)-1) ∧
    Exact (oldBits+bits 1) (old+1) ∧
    Exact (nextBits oldBits countBits) (RealControl.nextK old (count p)) := by
  have hn := count_lt p h
  have one : Exact (bits 1) 1 := bits_exact (by norm_num [Fits])
  have last := sub_exact hc one (by unfold Fits limit at *; omega)
  have inc := add_exact ho one (by unfold Fits limit at *; omega)
  refine ⟨last,inc,?_⟩
  unfold nextBits RealControl.nextK
  rw [eq_exact ho last]
  by_cases he : old = (count p : Int)-1
  · simpa [he] using (bits_exact (z := 0) (by norm_num [Fits]))
  · simpa [he] using inc

theorem previousK_bounds (p : Parameters) (i : Nat) (hk : 0 < count p) :
    -1 ≤ RealLoopState.previousK p i ∧
      RealLoopState.previousK p i < (count p : Int) := by
  by_cases hi : i = 0
  · simp only [RealLoopState.previousK,hi,if_true]
    omega
  · have hm := Nat.mod_lt (i-1) hk
    change (i-1)%RealLoopState.count p < count p at hm
    simp only [RealLoopState.previousK,if_neg hi]
    omega

/-- For every real flat-loop iteration, the actual i32 update decodes to
the proved mathematical loop phase. This includes the very first iteration. -/
theorem phase_exact (p : Parameters) (h : Admissible p) (i : Nat)
    (hk : 0 < count p) (oldBits countBits : BitVec 32)
    (ho : Exact oldBits (RealLoopState.previousK p i)) (hc : Exact countBits (count p)) :
    Exact (nextBits oldBits countBits) (RealLoopState.phase p i) := by
  have hh := (next_exact p h _ (previousK_bounds p i hk) oldBits countBits ho hc).2.2
  change Exact (nextBits oldBits countBits)
    (RealControl.nextK (RealLoopState.previousK p i) (RealLoopState.count p)) at hh
  rw [RealLoopState.next_phase p i hk] at hh
  exact hh

/-- At a real K restart, adding SMS to the signed prior tile reaches the
correct persistent ordinal. No negative value is passed to div/rem. -/
theorem tile_restart_exact (p : Parameters) (h : Admissible p) (pid i : Nat)
    (hp : pid < p.SMS) (hk : 0 < count p) (hi : i < trip p pid)
    (hr : i % count p = 0) (oldBits : BitVec 32)
    (ho : Exact oldBits (FlatSchedule.previousTile p.SMS pid (count p) i)) :
    Exact (oldBits+bits (p.SMS : Int)) (Schedule.tileId p.SMS pid (i/count p)) := by
  have hq := (Schedule.iteration_bounds (count p) (work p pid) i hk hi).1
  have ht := tile_lt p h pid (i/count p) hp hq
  have hs := nat_exact (h.scalar p.SMS (by simp))
  have he := FlatSchedule.tile_restart p.SMS pid (count p) i hk hr
  rw [he] at ho
  have hb := nat_fits (lt_trans ht h.total_bound)
  have hh := add_exact ho hs (by simpa only [sub_add_cancel] using hb)
  simpa only [sub_add_cancel,Schedule.tileId] using hh

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeControl
