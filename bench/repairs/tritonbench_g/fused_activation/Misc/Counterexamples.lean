import bench.repairs.tritonbench_g.fused_activation.Proofs.Contract

namespace VeriTile.Bench.Repairs.FusedActivation.Counterexamples

theorem repaired_block_three : min (2048 : Nat) 3 = 3 ∧ block 3 = 4 := by
  decide +kernel

/-- The original i32 control becomes negative at the first high tile, even
when the constexpr extent comparison is widened to signed i64. -/
theorem original_high_pid :
    (TritonMul2Finite.offset32 2048 (2^20) 0).toInt = -(2^31 : Int) ∧
    ((TritonMul2Finite.offset32 2048 (2^20) 0).signExtend 64).slt
      (BitVec.ofNat 64 (2^31+1)) = Bool.true := by decide +kernel

#axiomsClean repaired_block_three
#axiomsClean original_high_pid
end VeriTile.Bench.Repairs.FusedActivation.Counterexamples
