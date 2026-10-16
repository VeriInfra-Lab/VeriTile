import bench.tritonbench_g.matmul_persistent_triton.Proofs.HintMath
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeIteration

/-! Coordinate promises at actual reached iterations of the previously checked
signed-i32 source/control projection. The native compiler/backend remains an
external boundary. No execution or coordinate invariant is assumed. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintReached
open NativeDomain NativeBits HintMath
set_option maxHeartbeats 3000000

def tile (p : Parameters) (pid i : Nat) := Schedule.tileId p.SMS pid (i/count p)
def rowIndex (p : Parameters) (pid i : Nat) :=
  Schedule.row (rows p) (columns p) p.GM (tile p pid i)
def columnIndex (p : Parameters) (pid i : Nat) :=
  Schedule.column (rows p) (columns p) p.GM (tile p pid i)

def rowBits (p : Parameters) (pid i lane : Nat) : BitVec 32 :=
  NativeAddresses.clampBits (NativeAddresses.laneBits
    (NativeCoordinates.run p pid (i+1)).row p.BM lane) p.M
def columnBits (p : Parameters) (pid i lane : Nat) : BitVec 32 :=
  NativeAddresses.clampBits (NativeAddresses.laneBits
    (NativeCoordinates.run p pid (i+1)).column p.BN lane) p.N

def rowValues (p : Parameters) (pid i lane : Nat) : Int := (rowBits p pid i lane).toInt
def columnValues (p : Parameters) (pid i lane : Nat) : Int := (columnBits p pid i lane).toInt

theorem index_bounds (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    rowIndex p pid i < rows p ∧ columnIndex p pid i < columns p :=
  (Schedule.grouped_bounds (rows p) (columns p) p.GM _ h.gm
    (NativeCoordinates.current_tile_bound p h pid hp hk i hi)).2

theorem row_eq (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (lane : Nat) (hl : lane < p.BM) :
    rowValues p pid i lane = clamped p.M p.BM (rowIndex p pid i) lane := by
  have hc := (NativeCoordinates.agrees p h pid hp hk i hi).1
  have hb := (index_bounds p h pid hp hk i hi).1
  have hx := (NativeAddresses.lane_exact _ p.BM lane _ hc
    (h.scalar _ (by simp)) hl (row_lane_lt p h _ lane hb hl)).2
  exact NativeAddresses.clamp_exact _ p.M _ hx (h.scalar _ (by simp))

theorem column_eq (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (lane : Nat) (hl : lane < p.BN) :
    columnValues p pid i lane = clamped p.N p.BN (columnIndex p pid i) lane := by
  have hc := (NativeCoordinates.agrees p h pid hp hk i hi).2
  have hb := (index_bounds p h pid hp hk i hi).2
  have hx := (NativeAddresses.lane_exact _ p.BN lane _ hc
    (h.scalar _ (by simp)) hl (column_lane_lt p h _ lane hb hl)).2
  exact NativeAddresses.clamp_exact _ p.N _ hx (h.scalar _ (by simp))

/-- First-lane alignment also holds in partial M/N tiles. Contiguity is a
separate property and is not inferred from this alignment theorem. -/
theorem reached_aligned (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    Aligned p.BM (rowValues p pid i) ∧ Aligned p.BN (columnValues p pid i) := by
  constructor
  · unfold Aligned
    rw [row_eq p h pid hp hk i hi 0 h.bm]
    exact clamped_aligned _ _ _
  · unfold Aligned
    rw [column_eq p h pid hp hk i hi 0 h.bn]
    exact clamped_aligned _ _ _

/-- BM divides M and BN divides N are explicit sufficient mathematical
conditions. K is unrestricted here: it may have a partial final block. -/
theorem reached_promises (p : Parameters) (h : Admissible p)
    (hm : p.BM ∣ p.M) (hn : p.BN ∣ p.N) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    TilePromises p.BM (rowValues p pid i) ∧ TilePromises p.BN (columnValues p pid i) := by
  have hb := index_bounds p h pid hp hk i hi
  exact ⟨promises_congr _ _ _ (full_tile_promises _ _ _ h.bm hm hb.1)
      (row_eq p h pid hp hk i hi),
    promises_congr _ _ _ (full_tile_promises _ _ _ h.bn hn hb.2)
      (column_eq p h pid hp hk i hi)⟩

/-- Every K phase of the same persistent output tile has the same clamped
coordinate vector, including the last partial K phase. -/
theorem same_tile_values (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i j : Nat)
    (hi : i < trip p pid) (hj : j < trip p pid) (he : i/count p = j/count p) :
    (∀ lane, lane < p.BM → rowValues p pid i lane = rowValues p pid j lane) ∧
    (∀ lane, lane < p.BN → columnValues p pid i lane = columnValues p pid j lane) := by
  constructor
  · intro lane hl
    rw [row_eq p h pid hp hk i hi lane hl,row_eq p h pid hp hk j hj lane hl]
    simp only [rowIndex,tile,he]
  · intro lane hl
    rw [column_eq p h pid hp hk i hi lane hl,column_eq p h pid hp hk j hj lane hl]
    simp only [columnIndex,tile,he]

/-- Any reached partially active row tile violates the full-block consecutive
coordinate assertion. The condition refers to the derived grouped schedule. -/
theorem partial_row_not_contiguous (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (tail : Nat) (ht : p.M = rowIndex p pid i * p.BM+tail)
    (hpos : 0 < tail) (hlt : tail < p.BM) :
    ¬ Contiguous p.BM (rowValues p pid i) := by
  intro hc
  apply partial_tile_not_contiguous p.M p.BM (rowIndex p pid i) tail ht hpos hlt
  intro lane hl
  have hh := hc lane hl
  rwa [row_eq p h pid hp hk i hi lane hl,row_eq p h pid hp hk i hi 0 h.bm] at hh

theorem partial_column_not_contiguous (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (tail : Nat) (ht : p.N = columnIndex p pid i * p.BN+tail)
    (hpos : 0 < tail) (hlt : tail < p.BN) :
    ¬ Contiguous p.BN (columnValues p pid i) := by
  intro hc
  apply partial_tile_not_contiguous p.N p.BN (columnIndex p pid i) tail ht hpos hlt
  intro lane hl
  have hh := hc lane hl
  rwa [column_eq p h pid hp hk i hi lane hl,column_eq p h pid hp hk i hi 0 h.bn] at hh

theorem zero_K_no_reached_hint (p : Parameters) (h : Admissible p) (hz : p.K = 0)
    (pid i : Nat) : ¬ i < trip p pid := by
  have hb := h.bk
  have hc : count p = 0 := by
    unfold count RealMath.blocks
    rw [hz]
    simp only [Nat.zero_add]
    exact Nat.div_eq_of_lt (by omega)
  simp [trip,hc]

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintReached
