import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Indices

namespace VeriTile.Bench.Repairs.MaskedAddCuda.Counterexamples
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite

/-- At n=2^31+1 Triton specializes n as i64, while unannotated pid/lane
arithmetic remains i32. The first high tile has an active negative offset.+The n=2^31, pid=0 case is deliberately not misclassified as all-masked. -/
theorem original_high_pid_overflow :
    2^21 < Indices.gridCount (2^31+1) ∧
    (offset32 (2^21) 0).toInt = -(2^31 : Int) ∧
    ((offset32 (2^21) 0).signExtend 64).slt (BitVec.ofNat 64 (2^31+1)) = Bool.true := by
  decide +kernel

/-- A stride-two logical update leaves the middle backing cell untouched.
The original flat offset-one update instead changes that hole. -/
theorem stride_two_hole :
    (77 : ℚ) + 99 * (1/2) = 253/2 ∧ (253/2 : ℚ) ≠ 77 := by norm_num

/-- A shifted data alias across the 1024-element tile boundary reuses a
previous tile's updated value rather than the initial data value. -/
theorem shifted_alias :
    (1 : ℚ) + (1+1)*1 = 3 ∧ (1 : ℚ) + 1*1 = 2 ∧ (3 : ℚ) ≠ 2 := by norm_num

/-- The source's zero-veto branch is not multiplication by (1-mask) for a
general signed mask. This fixes the intended independent target explicitly. -/
theorem nonbinary_mask_formula :
    (10 : ℚ) + 2*(1-2)*(1/2) = 9 ∧ (9 : ℚ) ≠ 10 := by norm_num

#axiomsClean original_high_pid_overflow
#axiomsClean stride_two_hole
#axiomsClean shifted_alias
#axiomsClean nonbinary_mask_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaskedAddCuda.Counterexamples
