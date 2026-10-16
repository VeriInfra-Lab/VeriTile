import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealGroup
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealControl
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStep
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealPrefix
import bench.tritonbench_g.matmul_persistent_triton.Proofs.FlatSchedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoopState
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def count (p : Parameters) := RealMath.blocks p.K p.BK
def work (p : Parameters) (pid : Nat) :=
  Schedule.tilesFor (RealGroup.rows p*RealGroup.columns p) p.SMS pid
def stop (p : Parameters) (pid : Nat) := count p * work p pid
def phase (p : Parameters) (i : Nat) := i % count p
def tile (p : Parameters) (pid i : Nat) := Schedule.tileId p.SMS pid (i/count p)
def row (p : Parameters) (pid i : Nat) :=
  Schedule.row (RealGroup.rows p) (RealGroup.columns p) p.GM (tile p pid i)
def column (p : Parameters) (pid i : Nat) :=
  Schedule.column (RealGroup.rows p) (RealGroup.columns p) p.GM (tile p pid i)
def previousK (p : Parameters) (i : Nat) : Int :=
  if i=0 then -1 else ((i-1)%count p : Nat)
def accumulator (p : Parameters) (xs : RealMath.Inputs) (pid i : Nat) :=
  RealDot.accumulator p xs (RealStore.rows p (row p pid i))
    (RealStore.columns p (column p pid i)) (phase p i)

/-- Registers initialized by the real prefix and unchanged by every loop body. -/
structure Fixed (p : Parameters) (s : BlockState) : Prop where
  count : s.regs .nat [] "k_tiles" = some (Tile.scalar (count p))
  rows : s.regs .nat [] "num_pid_m" = some (Tile.scalar (RealGroup.rows p))
  group : s.regs .nat [] "num_pid_in_group" = some (Tile.scalar (p.GM*RealGroup.columns p))
  lanes : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)

structure Coordinates (p : Parameters) (pid i : Nat) (s : BlockState) : Prop where
  row : s.regs .int [] "pid_m" = some (Tile.scalar (row p pid i : Int))
  column : s.regs .int [] "pid_n" = some (Tile.scalar (column p pid i : Int))
  a : s.regs .int [p.BM] "offs_am" = some (Tile.vec fun j =>
    (RealTiles.clamp p.M (RealStore.rows p (RealLoopState.row p pid i) j) : Int))
  b : s.regs .int [p.BN] "offs_bn" = some (Tile.vec fun j =>
    (RealTiles.clamp p.N (RealStore.columns p (RealLoopState.column p pid i) j) : Int))

/-- The invariant is indexed by the original flat source iteration. At a K
restart, the accumulator is already zero while coordinates may still describe
the preceding tile; the source setup, rather than a premise, replaces them. -/
structure Holds (p : Parameters) (xs : RealMath.Inputs) (pid i : Nat) (s : BlockState) : Prop where
  fixed : Fixed p s
  ki : s.regs .int [] "ki" = some (Tile.scalar (previousK p i))
  tile : s.regs .int [] "tile_id" = some (Tile.scalar (FlatSchedule.previousTile p.SMS pid (count p) i))
  acc : s.regs .real [p.BM,p.BN] "accumulator" = some (accumulator p xs pid i)
  coords : phase p i ≠ 0 → Coordinates p pid i s

theorem next_phase (p : Parameters) (i : Nat) (hk : 0<count p) :
    RealControl.nextK (previousK p i) (count p) = (phase p i : Int) := by
  cases i with
  | zero => simpa [previousK,phase] using RealControl.initial (count p) hk
  | succ i => simpa [previousK,phase,Nat.succ_eq_add_one] using RealControl.next (count p) i hk

theorem tile_bound (p : Parameters) (pid i : Nat) (hk : 0<count p)
    (hs : 0<p.SMS) (hp : pid<p.SMS) (hi : i<stop p pid) :
    tile p pid i < RealGroup.rows p*RealGroup.columns p := by
  apply (Schedule.ordinal_bound_iff _ _ _ _ hs hp).mp
  exact (Schedule.iteration_bounds (count p) (work p pid) i hk hi).1

theorem next_tile_register (p : Parameters) (pid i : Nat) :
    FlatSchedule.previousTile p.SMS pid (count p) (i+1) = (tile p pid i : Int) := by
  simp [FlatSchedule.previousTile,tile,Schedule.tileId]

theorem continue_phase (p : Parameters) (i : Nat) (hk : 0<count p)
    (h : phase p i+1 ≠ count p) : phase p (i+1) = phase p i+1 := by
  have hm := Nat.mod_lt i hk
  simp only [phase, ← Nat.mod_add_mod i (count p) 1]
  exact Nat.mod_eq_of_lt (by dsimp [phase] at h; omega)

theorem continue_tile (p : Parameters) (pid i : Nat) (hk : 0<count p)
    (h : phase p i+1 ≠ count p) : tile p pid (i+1) = tile p pid i := by
  unfold tile
  rw [FlatSchedule.quotient_continue (count p) i hk h]

theorem initial (p : Parameters) (xs : RealMath.Inputs) (s : BlockState) :
    Holds p xs (s.pids 0) 0 (RealPrefix.result p s) := by
  have hf : Fixed p (RealPrefix.result p s) := by
    constructor <;> by_cases hd : s.pids 0 < RealPrefix.total p % p.SMS <;>
      simp [RealPrefix.result,hd,count,RealPrefix.kTiles,RealPrefix.rows,
        RealPrefix.columns,RealGroup.rows,RealGroup.columns]
  refine ⟨hf, ?_, ?_, ?_, ?_⟩
  · simp [RealPrefix.result,previousK]
  · simp [RealPrefix.result,FlatSchedule.previousTile]
  · simp [RealPrefix.result,accumulator,RealDot.accumulator,phase,RealMath.partial_zero]
  · simp [phase]

theorem set_index (p : Parameters) (xs : RealMath.Inputs) (pid i : Nat) (s : BlockState)
    (h : Holds p xs pid i s) : Holds p xs pid i (s.setReg "_i" .nat [] (Tile.scalar i)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact ⟨by simpa using h.fixed.count, by simpa using h.fixed.rows,
      by simpa using h.fixed.group, by simpa using h.fixed.lanes⟩
  · simpa using h.ki
  · simpa using h.tile
  · simpa using h.acc
  · intro hi
    exact ⟨by simpa using (h.coords hi).row, by simpa using (h.coords hi).column,
      by simpa using (h.coords hi).a, by simpa using (h.coords hi).b⟩

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoopState
