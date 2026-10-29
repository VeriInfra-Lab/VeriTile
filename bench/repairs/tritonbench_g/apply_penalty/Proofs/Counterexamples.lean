import bench.repairs.tritonbench_g.apply_penalty.Proofs.Contract
import bench.repairs.tritonbench_g.apply_penalty.Kernels

namespace VeriTile.Bench.Repairs.ApplyPenalty.Counterexamples

/-- Original int32 start+lane wraps on an empty high-offset row and falsely
enables its upper-bound mask. Widening before the addition prevents this. -/
theorem original_mask_wrap :
    (BitVec.ofNat 32 (2^31-1)+1).slt (BitVec.ofNat 32 (2^31-1)) = Bool.true ∧
    ¬ ((2^31-1 : Nat)+1 < 2^31-1) := by decide

theorem repaired_mask_no_wrap :
    (BitVec.ofNat 64 (2^31-1)+1).slt (BitVec.ofNat 64 (2^31-1)) = Bool.false := by decide

theorem understated_hint_misses_record : sourceBlock 512=512 ∧ ¬ (512 : Nat)<sourceBlock 512 := by
  decide

theorem original_demo_cumsum_short : ¬ (2+1 : Nat)≤2 := by decide

/-- Incompatible duplicate records cannot both specify the final same cell.
Equal-count duplicates are also rejected: same-value GPU writes still race. -/
theorem duplicate_targets_conflict : penalty 8 1 0 1 2 ≠ penalty 8 2 0 1 2 := by
  norm_num [penalty]

#axiomsClean original_mask_wrap
#axiomsClean repaired_mask_no_wrap
#axiomsClean understated_hint_misses_record
#axiomsClean original_demo_cumsum_short
#axiomsClean duplicate_targets_conflict
end VeriTile.Bench.Repairs.ApplyPenalty.Counterexamples
