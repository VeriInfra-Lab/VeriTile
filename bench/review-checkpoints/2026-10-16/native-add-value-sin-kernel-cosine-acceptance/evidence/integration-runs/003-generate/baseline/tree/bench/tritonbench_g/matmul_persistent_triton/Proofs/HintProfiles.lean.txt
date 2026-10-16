import bench.tritonbench_g.matmul_persistent_triton.Proofs.HintReached
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeConfig

/-! Both recorded native compile profiles have dynamic signed-i32 M/N/K.
These witnesses instantiate those same compile constants; they are not new
native execution receipts or assumptions of backend soundness. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintProfiles
open NativeDomain NativeBits NativeConfig HintMath HintReached
set_option maxHeartbeats 4000000

theorem witness_count_pos (storage : TypedSource.Storage) : 0 < count (witness storage) := by
  cases storage <;> norm_num [count,witness,blockK,RealMath.blocks]

/-- Every reached iteration, including repeated output tiles and partial K,
has aligned consecutive coordinate vectors in the already recorded witness. -/
theorem witness_reached_promises (storage : TypedSource.Storage) (pid i : Nat)
    (hp : pid < grid (witness storage)) (hi : i < trip (witness storage) pid) :
    TilePromises 128 (rowValues (witness storage) pid i) ∧
    TilePromises 256 (columnValues (witness storage) pid i) :=
  reached_promises (witness storage) (witness_admissible storage)
    (by change 128 ∣ 2176; decide) (by change 256 ∣ 2304; decide)
    pid hp (witness_count_pos storage) i hi

/-- The concrete last K phase has one active lane in both profiles, while
both M/N hint-coordinate obligations continue to hold. -/
theorem witness_partial_K (storage : TypedSource.Storage) :
    let p := witness storage
    let i := count p-1
    p.K % p.BK = 1 ∧
    i < trip p 0 ∧
    TilePromises 128 (rowValues p 0 i) ∧
    TilePromises 256 (columnValues p 0 i) ∧
    (bits 0).slt (NativeAddresses.tailBits p.K (NativeLoop.run p 0 (i+1)).ki p.BK) = Bool.true ∧
    (bits 1).slt (NativeAddresses.tailBits p.K (NativeLoop.run p 0 (i+1)).ki p.BK) = Bool.false := by
  have hp : 0 < grid (witness storage) := by
    rw [(witness_nonempty storage).1]
    decide
  have hi : count (witness storage)-1 < trip (witness storage) 0 := by
    have hc := witness_count_pos storage
    have hw := (witness_nonempty storage).2.1
    unfold trip
    rw [hw]
    omega
  have hprom := witness_reached_promises storage 0 _ hp hi
  refine ⟨?_,hi,hprom.1,hprom.2,?_,?_⟩
  · cases storage <;> norm_num [witness,blockK]
  · have ht := NativeIteration.load_tail_mask (witness storage) (witness_admissible storage)
      0 hp (witness_count_pos storage) _ hi 0 (by cases storage <;> simp [witness,blockK])
    convert ht using 1; cases storage <;> norm_num [count,witness,blockK,RealMath.blocks]
  · have ht := NativeIteration.load_tail_mask (witness storage) (witness_admissible storage)
      0 hp (witness_count_pos storage) _ hi 1 (by cases storage <;> simp [witness,blockK])
    convert ht using 1; cases storage <;> norm_num [count,witness,blockK,RealMath.blocks]

/-- A fully explicit runtime M/N-tail case inside the original signed-i32
arithmetic domain and both recorded compile profiles. No public domain is
narrowed: this is a counterexample to a separate full-contiguity assertion. -/
def tailWitness (storage : TypedSource.Storage) : Parameters :=
  { witness storage with M := 129, N := 257, sbk := 257, scm := 257 }

theorem tail_profile (storage : TypedSource.Storage) :
    RecordedCompileProfile (tailWitness storage) storage 8 (stages storage) := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem tail_admissible (storage : TypedSource.Storage) : Admissible (tailWitness storage) := by
  cases storage <;> constructor <;>
    norm_num [tailWitness,witness,blockK,limit,rows,columns,count,total,
      rowPad,columnPad,kPad,RealMath.blocks]

theorem tail_geometry (storage : TypedSource.Storage) :
    grid (tailWitness storage) = 4 ∧ work (tailWitness storage) 3 = 1 ∧
    0 < count (tailWitness storage) ∧
    (tailWitness storage).K % (tailWitness storage).BK = 1 := by
  cases storage <;> norm_num [tailWitness,witness,blockK,grid,work,total,rows,columns,count,
    RealMath.blocks,Schedule.tilesFor]

theorem tail_reached_bounds (storage : TypedSource.Storage) (i : Nat)
    (hi : i < count (tailWitness storage)) :
    3 < grid (tailWitness storage) ∧ i < trip (tailWitness storage) 3 := by
  obtain ⟨hg,hw,_⟩ := tail_geometry storage
  exact ⟨by omega,by simpa [trip,hw] using hi⟩

theorem tail_row_eq (storage : TypedSource.Storage) (i lane : Nat)
    (hi : i < count (tailWitness storage)) (hl : lane < 128) :
    rowValues (tailWitness storage) 3 i lane = clamped 129 128 1 lane := by
  have hb := tail_reached_bounds storage i hi
  rw [row_eq (tailWitness storage) (tail_admissible storage) 3 hb.1
    (tail_geometry storage).2.2.1 i hb.2 lane hl]
  simp only [rowIndex,tile,Nat.div_eq_of_lt hi]
  norm_num [tailWitness,witness,rows,columns,RealMath.blocks,
    Schedule.tileId,Schedule.row,Schedule.groupBase,Schedule.groupSize]

theorem tail_column_eq (storage : TypedSource.Storage) (i lane : Nat)
    (hi : i < count (tailWitness storage)) (hl : lane < 256) :
    columnValues (tailWitness storage) 3 i lane = clamped 257 256 1 lane := by
  have hb := tail_reached_bounds storage i hi
  rw [column_eq (tailWitness storage) (tail_admissible storage) 3 hb.1
    (tail_geometry storage).2.2.1 i hb.2 lane hl]
  simp only [columnIndex,tile,Nat.div_eq_of_lt hi]
  norm_num [tailWitness,witness,rows,columns,RealMath.blocks,
    Schedule.tileId,Schedule.column,Schedule.groupBase,Schedule.groupSize]

/-- Actual reached vectors are [128,0,...] and [256,0,...]. Their first
adjacent pair refutes the claimed full 128/256 consecutive-lane groups. -/
theorem tail_not_contiguous (storage : TypedSource.Storage) (i : Nat)
    (hi : i < count (tailWitness storage)) :
    ¬ Contiguous 128 (rowValues (tailWitness storage) 3 i) ∧
    ¬ Contiguous 256 (columnValues (tailWitness storage) 3 i) := by
  constructor
  · intro hc
    have hh := hc 1 (by decide)
    rw [tail_row_eq storage i 1 hi (by decide),tail_row_eq storage i 0 hi (by decide)] at hh
    norm_num [clamped,clamp] at hh
  · intro hc
    have hh := hc 1 (by decide)
    rw [tail_column_eq storage i 1 hi (by decide),tail_column_eq storage i 0 hi (by decide)] at hh
    norm_num [clamped,clamp] at hh

theorem tail_still_aligned (storage : TypedSource.Storage) (i : Nat)
    (hi : i < count (tailWitness storage)) :
    Aligned 128 (rowValues (tailWitness storage) 3 i) ∧
    Aligned 256 (columnValues (tailWitness storage) 3 i) := by
  have hb := tail_reached_bounds storage i hi
  exact reached_aligned _ (tail_admissible storage) 3 hb.1 (tail_geometry storage).2.2.1 i hb.2

/-- Both the first reached hint site and its last-K cached use fail full
contiguity. The K tail does not repair the M/N-tail coordinate vectors. -/
theorem tail_first_and_last (storage : TypedSource.Storage) :
    ¬ Contiguous 128 (rowValues (tailWitness storage) 3 0) ∧
    ¬ Contiguous 256 (columnValues (tailWitness storage) 3 0) ∧
    ¬ Contiguous 128 (rowValues (tailWitness storage) 3 (count (tailWitness storage)-1)) ∧
    ¬ Contiguous 256 (columnValues (tailWitness storage) 3 (count (tailWitness storage)-1)) := by
  have hk := (tail_geometry storage).2.2.1
  have hf := tail_not_contiguous storage 0 hk
  have hl := tail_not_contiguous storage _ (show count (tailWitness storage)-1 < count (tailWitness storage) by omega)
  exact ⟨hf.1,hf.2,hl.1,hl.2⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintProfiles
