import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Integer loop bounds for the descending source range. After the frontend
substitutes j = -i_t, j runs from 1-chunks to 1 with positive unit stride.
The number of remaining chunks includes the next chunk to be visited. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex
open VeriTile.Triton

def remaining (j : Int) : Nat := (1 - j).toNat

theorem at_start (chunks : Nat) : remaining (1 - (chunks : Int)) = chunks := by
  unfold remaining
  omega

theorem at_end : remaining 1 = 0 := by rfl

theorem at_or_after_end (j : Int) (hj : 1 ≤ j) : remaining j = 0 := by
  unfold remaining
  omega

theorem current_chunk (j : Int) (hj : j < 1) :
    remaining j = (-j).toNat + 1 := by
  unfold remaining
  omega

theorem after_step (j : Int) : remaining (j + 1) = (-j).toNat := by
  unfold remaining
  congr 1
  omega

theorem chunk_in_range (chunks : Nat) (j : Int)
    (hl : 1 - (chunks : Int) ≤ j) (hu : j < 1) :
    (-j).toNat < chunks := by
  omega

theorem initial_empty : ¬(1 - (0 : Int) < 1) := by omega

theorem index_nonnegative (j : Int) (hj : j < 1) : 0 ≤ -j := by omega

theorem int_index_of_nat (j : Int) (hj : j < 1) : ((-j).toNat : Int) = -j := by
  omega

/-- Signed block offsets are converted only after the loop bound establishes
that the actual source index is nonnegative. -/
theorem offset_toNat (j : Int) (BT : Nat) (hj : j < 1) :
    ((-j) * (BT : Int)).toNat = (-j).toNat * BT := by
  have hmul : (-j) * (BT : Int) = (((-j).toNat * BT : Nat) : Int) := by
    simp only [Int.natCast_mul, int_index_of_nat j hj]
  rw [hmul]
  rfl

/-- Regression witness for the previous natural-number loop-count lowering.
At zero chunks it incorrectly requested one iteration. The integer bounds
above request no iterations and match Python's range(-1,-1,-1). -/
theorem previous_nat_empty_boundary :
    ((0 : Nat) - 1) / 1 + 1 = 1 ∧ remaining 1 = 0 := by
  decide

#axiomsClean at_start
#axiomsClean at_or_after_end
#axiomsClean current_chunk
#axiomsClean after_step
#axiomsClean chunk_in_range
#axiomsClean initial_empty
#axiomsClean offset_toNat
#axiomsClean previous_nat_empty_boundary
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.SignedIndex
