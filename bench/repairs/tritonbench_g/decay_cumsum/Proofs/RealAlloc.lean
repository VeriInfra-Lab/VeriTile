import bench.repairs.tritonbench_g.decay_cumsum.Proofs.PackedStorage
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealTiles

/-! The original Real proofs use FlatAlloc's finite cell-space translation.
This is a verification representation of the same named packed views; physical
byte addresses remain PackedStorage.address and NativeIndex.source_addresses.
The translation is not a claim that one scalar occupies one native byte.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.RealAlloc
open VeriTile.Triton Geometry Grid PackedStorage
noncomputable section
set_option maxHeartbeats 4000000

/-- A finite verifier cell-space image of the actual allocated regions. -/
def allocation (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase) : FlatAlloc where
  flat := "decay_packed_flat"
  regions := Roles.buffers kind
  base := fun r => (storage.allocation r).buffer.base+shift h p
  extent := fun r => if r∈Roles.buffers kind then h.cells-shift h p else 0

theorem disjoint (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase) :
    (allocation h kind storage p).Disjoint := by
  intro r hr s hs different
  change r ∈ Roles.buffers kind at hr
  change s ∈ Roles.buffers kind at hs
  have sep := storage.separate r hr s hs different
  have er := storage.extent r hr
  have es := storage.extent s hs
  have bound := Nat.sub_le h.cells (shift h p)
  dsimp [allocation,FlatAlloc.Disjoint]
  rw [if_pos hr,if_pos hs]
  rcases sep with sep|sep
  · left
    rw [er] at sep
    omega
  · right
    rw [es] at sep
    omega

theorem extent_bound (h : Call) (kind : Kind) (storage : Placement h kind)
    (w : Work h) (member : w∈schedule h) (r : RegionName) (hr : r∈Roles.buffers kind)
    (t : Fin w.1.span) (j : Fin h.BK) :
    relative h w.1 w.2 t.val j.val<(allocation h kind storage w.1).extent r := by
  have bounds := offset_inside h w.1 (schedule_valid h w member) w.2 t j
  rw [←relocated] at bounds
  simp only [allocation,if_pos hr]
  omega

theorem feature_positive (h : Call) (p : Phase) (prog : Program h p) : 0<h.DK := by
  by_contra hn
  have zero : h.DK=0 := by omega
  have features : h.features=0 := by
    unfold Call.features
    rw [zero]
    simp only [Nat.zero_add]
    exact Nat.div_eq_of_lt (Nat.sub_lt h.featureTilePositive (by decide))
  have bound := prog.featureInside
  rw [features] at bound
  omega

def entry (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (heap : Heap ℝ) : BlockState where
  mem := fun r o => .real (lift storage w.1 heap r o)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then w.2.feature else if ax=1 then w.2.chunk else w.2.row
  numPids := fun ax => if ax=0 then h.features else if ax=1 then w.1.count else h.rows
  undef := fun _ _ => 0

/-- Pulling a completed flat source result back to the same named view. -/
def pull (h : Call) (kind : Kind) (storage : Placement h kind) (p : Phase) (v : BlockState) : RegionName → Nat → ℝ :=
  fun r o => v.readMem (allocation h kind storage p).flat ((allocation h kind storage p).addr r o)

theorem pull_entry (h : Call) (kind : Kind) (storage : Placement h kind) (w : Work h) (heap : Heap ℝ)
    (r : RegionName) (hr : r∈Roles.buffers kind) (o : Nat) (ho : shift h w.1+o<h.cells) :
    pull h kind storage w.1 ((allocation h kind storage w.1).flattenState (entry h kind storage w heap)) r o=
      lift storage w.1 heap r o := by
  apply FlatAlloc.flattenState_readMem
  · exact disjoint h kind storage w.1
  · exact hr
  · simp only [allocation,if_pos hr]
    omega

#axiomsClean extent_bound
#axiomsClean pull_entry
end
end VeriTile.Bench.Repairs.DecayCumsum.RealAlloc
