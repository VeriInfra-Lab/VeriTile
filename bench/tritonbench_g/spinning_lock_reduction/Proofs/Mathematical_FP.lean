import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Fresh
import bench.tritonbench_g.spinning_lock_reduction.Proofs.SpinningLock_ConcurrentCorrect_Mathematical_ConcurrentCorrect
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.TritonBenchG.SpinningLock
open VeriTile.Triton

/-- Exact +0 + +0 in the selected closed FP32 software semantics. -/
theorem fp32_positive_zero_add : fp32.add fp32.zero fp32.zero = fp32.zero := by
  simpa only [fp32] using FP.ScalarOpsLaws.add_positive_zero_fp32

theorem fp32_positive_zero_value : fp32.zero.toRat? = some 0 := by
  simpa only [fp32] using FP.ScalarOpsLaws.positive_zero_value_fp32

theorem fresh_fp32 (h : HostArgs) (buffers : HostBuffers h) (schedule : List Nat)
    (mem : Memory (FP.Value .fp32)) (fresh : ∀ a, mem.locks a = 0) :
    ∃ result, FreshFacts fp32 h buffers schedule mem result :=
  fresh_source_backend fp32 fp32_positive_zero_add h buffers schedule mem fresh

#axiomsClean fp32_positive_zero_add
#axiomsClean fresh_fp32
end VeriTile.Bench.TritonBenchG.SpinningLock
