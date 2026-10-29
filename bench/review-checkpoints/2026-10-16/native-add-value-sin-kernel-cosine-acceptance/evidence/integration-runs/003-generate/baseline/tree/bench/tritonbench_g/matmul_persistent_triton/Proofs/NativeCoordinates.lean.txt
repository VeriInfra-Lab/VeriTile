import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeLoop
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeGroup
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeAddresses

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeCoordinates
open NativeDomain NativeBits
set_option maxHeartbeats 4000000

structure Coordinates where
  row : BitVec 32
  column : BitVec 32

/-- Source cached coordinates: recompute only after the actual i32 K-update
equals zero, retaining the previous coordinates at every other K iteration. -/
def run (p : Parameters) (pid : Nat) : Nat → Coordinates
  | 0 => ⟨bits 0,bits 0⟩
  | i+1 =>
    let s := NativeLoop.run p pid (i+1)
    if s.ki == bits 0 then
      let g := NativeGroup.eval p (NativePrefix.eval p pid).nr
        (NativePrefix.eval p pid).groupSpan s.tile
      ⟨g.row,g.column⟩
    else run p pid i

def Agrees (p : Parameters) (pid i : Nat) (c : Coordinates) : Prop :=
  Exact c.row (Schedule.row (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))) ∧
  Exact c.column (Schedule.column (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p)))

theorem control_after (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    Exact (NativeLoop.run p pid (i+1)).tile (Schedule.tileId p.SMS pid (i/count p)) ∧
    Exact (NativeLoop.run p pid (i+1)).ki ((i%count p : Nat) : Int) := by
  have ha := NativeLoop.agrees p h pid hp hk (i+1) (by omega)
  simpa only [FlatSchedule.previousTile,Nat.add_eq_zero_iff,show ¬(1:Nat)=0 by decide,
    and_false,if_false,Nat.add_sub_cancel,Schedule.tileId,RealLoopState.previousK] using ha

theorem current_tile_bound (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    Schedule.tileId p.SMS pid (i/count p) < total p :=
  tile_lt p h pid (i/count p) (lt_of_lt_of_le hp (Nat.min_le_left _ _))
    (Schedule.iteration_bounds (count p) (work p pid) i hk hi).1

theorem restart (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (hr : i%count p = 0) : Agrees p pid i (run p pid (i+1)) := by
  have hctrl := control_after p h pid hp hk i hi
  have hprefix := NativePrefix.agrees p h pid hp
  have hg := NativeGroup.agrees p h _ (current_tile_bound p h pid hp hk i hi)
    _ _ _ hprefix.nr hprefix.groupSpan hctrl.1
  have he := eq_exact hctrl.2 (bits_exact (z := 0) (by norm_num [Fits]))
  simp only [hr,Int.natCast_zero,decide_true] at he
  change Agrees p pid i (if (NativeLoop.run p pid (i+1)).ki == bits 0 then _ else _)
  rw [he]
  exact ⟨hg.row,hg.column⟩

/-- All cached native i32 row/column values are derived from the original
prefix and loop, with no prepared-register or whole-execution premise. -/
theorem agrees (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    Agrees p pid i (run p pid (i+1)) := by
  induction i with
  | zero => exact restart p h pid hp hk 0 hi (by simp)
  | succ i ih =>
    by_cases hr : (i+1)%count p = 0
    · exact restart p h pid hp hk (i+1) hi hr
    · have hctrl := control_after p h pid hp hk (i+1) hi
      have he := eq_exact hctrl.2 (bits_exact (z := 0) (by norm_num [Fits]))
      simp only [Int.natCast_eq_zero,hr,decide_false] at he
      have hlast : i%count p+1 ≠ count p := by
        intro hh
        exact hr ((FlatSchedule.last_iff_restart (count p) i hk).mp hh)
      have hquot := FlatSchedule.quotient_continue (count p) i hk hlast
      have hprev := ih (by omega)
      change Agrees p pid (i+1) (if (NativeLoop.run p pid (i+1+1)).ki == bits 0 then _ else _)
      rw [he]
      simpa only [Bool.false_eq_true,if_false,Agrees,hquot] using hprev

/-- The actual cached coordinates and K-control projection feed the original
A/B address expressions. Every lane, including masked tails, has the input-only
mathematical source offset under the explicit arithmetic domain. -/
theorem load_addresses (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (rl cl kl : Nat) (hrl : rl < p.BM) (hcl : cl < p.BN) (hkl : kl < p.BK) :
    let row := Schedule.row (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let col := Schedule.column (rows p) (columns p) p.GM (Schedule.tileId p.SMS pid (i/count p))
    let ki := i%count p
    let coords := run p pid (i+1)
    let kb := (NativeLoop.run p pid (i+1)).ki
    Exact (NativeAddresses.offsetBits (NativeAddresses.clampBits
      (NativeAddresses.laneBits coords.row p.BM rl) p.M)
      (NativeAddresses.laneBits kb p.BK kl) p.sam p.sak)
      (clamp p.M (row*p.BM+rl)*p.sam+(ki*p.BK+kl)*p.sak) ∧
    Exact (NativeAddresses.offsetBits (NativeAddresses.laneBits kb p.BK kl)
      (NativeAddresses.clampBits (NativeAddresses.laneBits coords.column p.BN cl) p.N) p.sbk p.sbn)
      ((ki*p.BK+kl)*p.sbk+clamp p.N (col*p.BN+cl)*p.sbn) := by
  have hcoords := agrees p h pid hp hk i hi
  have hki := (control_after p h pid hp hk i hi).2
  have hg := Schedule.grouped_bounds (rows p) (columns p) p.GM _ h.gm
    (current_tile_bound p h pid hp hk i hi)
  have hphase := Nat.mod_lt i hk
  exact ⟨NativeAddresses.a_exact p h _ _ rl kl hg.2.1 hphase hrl hkl _ _ hcoords.1 hki,
    NativeAddresses.b_exact p h _ _ cl kl hg.2.2 hphase hcl hkl _ _ hcoords.2 hki⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NativeCoordinates
