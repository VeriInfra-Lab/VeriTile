import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealKeys
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Defect
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
/-- Signed INT32 result after the original upper-bound additions. -/
def oldUpperNumerator : BitVec 32 := (63 : BitVec 32) + 2147483647 + 64

theorem old_upper_negative : oldUpperNumerator.toInt = -2147483522 := by decide

theorem only_kept_key :
    RealKeys.keep 0 64 0 1 0 2147483647 2147483647 0 = false ∧
    RealKeys.keep 0 64 0 1 0 2147483647 2147483647 1 = true := by decide

/-- Truncating signed division used by Triton, followed by the block product. -/
theorem old_loop_skips : oldUpperNumerator.toInt.tdiv 64 * 64 = -2147483520 := by decide

theorem promoted_upper_contains_key :
    min (Int.tdiv (63 + 2147483647 + 64) 64 * 64) 2 = 2 := by decide

/-- This separate large-workspace arithmetic observation does not allocate
or claim to execute a 128 GiB output. It explains the second PID promotion. -/
theorem old_carry_base_wraps :
    ((65534 : BitVec 32) * 32832).toInt = -2143355008 := by decide

theorem promoted_carry_base : (65534 : Int) * 32832 = 2151612288 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Defect
