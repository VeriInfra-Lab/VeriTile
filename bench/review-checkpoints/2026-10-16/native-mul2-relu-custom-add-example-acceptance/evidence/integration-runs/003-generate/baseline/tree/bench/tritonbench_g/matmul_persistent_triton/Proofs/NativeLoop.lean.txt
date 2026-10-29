import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativePrefix
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeControl

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeLoop
open NativeDomain NativeBits
set_option maxHeartbeats 4000000

structure Registers where
  tile : BitVec 32
  ki : BitVec 32

def step (sms count : BitVec 32) (s : Registers) : Registers :=
  let ki := NativeControl.nextBits s.ki count
  ⟨if ki == bits 0 then s.tile+sms else s.tile,ki⟩

/-- Execute exactly the source's integer control update, starting from its
actual negative initial values. This projects only tile/ki control, not GPU
memory, dot operations or the full native machine state. -/
def run (p : Parameters) (pid : Nat) : Nat → Registers
  | 0 => ⟨(NativePrefix.eval p pid).tile,(NativePrefix.eval p pid).ki⟩
  | i+1 => step (bits (p.SMS : Int)) (NativePrefix.eval p pid).nk (run p pid i)

theorem agrees (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i ≤ trip p pid) :
    Exact (run p pid i).tile (FlatSchedule.previousTile p.SMS pid (count p) i) ∧
    Exact (run p pid i).ki (RealLoopState.previousK p i) := by
  have hprefix := NativePrefix.agrees p h pid hp
  have hp' : pid < p.SMS := lt_of_lt_of_le hp (Nat.min_le_left _ _)
  induction i with
  | zero =>
    exact ⟨hprefix.tile,hprefix.ki⟩
  | succ i ih =>
    obtain ⟨htile,hki⟩ := ih (by omega)
    have hnext := NativeControl.phase_exact p h i hk _ _ hki hprefix.nk
    have heq := eq_exact hnext (bits_exact (z := 0) (by norm_num [Fits]))
    have htile' : Exact
        (if NativeControl.nextBits (run p pid i).ki (NativePrefix.eval p pid).nk == bits 0
          then (run p pid i).tile+bits (p.SMS : Int) else (run p pid i).tile)
        (Schedule.tileId p.SMS pid (i/count p)) := by
      rw [heq]
      by_cases hr : i%count p = 0
      · have hh := NativeControl.tile_restart_exact p h pid i hp' hk (by omega) hr _ htile
        have hr' : RealLoopState.phase p i = 0 := hr
        simpa only [hr',Int.natCast_zero,decide_true,if_true] using hh
      · have hh := FlatSchedule.tile_continue p.SMS pid (count p) i hk hr
        rw [hh] at htile
        have hr' : RealLoopState.phase p i ≠ 0 := hr
        simpa only [Int.natCast_eq_zero,hr',decide_false,Bool.false_eq_true,if_false,
          Schedule.tileId] using htile
    constructor
    · simpa only [run,step,FlatSchedule.previousTile,Nat.succ_ne_zero,if_false,
        Nat.succ_sub_one,Schedule.tileId] using htile'
    · simpa only [run,step,RealLoopState.previousK,Nat.succ_ne_zero,if_false,
        Nat.succ_sub_one,RealLoopState.phase] using hnext

/-- The complete control projection includes every original loop iteration;
its decoded sequence is derived by induction, not assumed as an invariant. -/
theorem all_iterations (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) :
    ∀ i, i < trip p pid →
      Exact (NativeControl.nextBits (run p pid i).ki (NativePrefix.eval p pid).nk)
        (RealLoopState.phase p i) := by
  intro i hi
  exact NativeControl.phase_exact p h i hk _ _ (agrees p h pid hp hk i hi.le).2
    (NativePrefix.agrees p h pid hp).nk

/-- Empty K gives an exact zero trip count, so the integer loop performs no
iteration. This agrees with the separately proved source memory-preservation
result; it does not fabricate output stores. -/
theorem zero_trip (p : Parameters) (h : Admissible p) (pid : Nat) (hK : p.K = 0) :
    count p = 0 ∧ trip p pid = 0 := by
  have hn : count p = 0 := by
    unfold count RealMath.blocks
    rw [hK,Nat.zero_add]
    exact Nat.div_eq_of_lt (by have hb := h.bk; omega)
  exact ⟨hn,by simp only [trip,hn,Nat.zero_mul]⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeLoop
