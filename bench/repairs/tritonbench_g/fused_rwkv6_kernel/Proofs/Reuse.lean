import bench.tritonbench_g.fused_rwkv6_kernel.FPCorrect
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Storage

/- The existing original-stage theorems remain available unchanged. Their
scope is an individual materialized state/output stage, not the repaired
whole register loop. Native exp remains a separate local primitive boundary.
The new independent loop composition extends coverage to reverse, rounded
small tiles, split-key host sum, the repaired output pitch and zero time. -/
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Reuse
#axiomsClean VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect.state_fp32_stage
#axiomsClean VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect.output_fp32_stage
#axiomsClean VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect.conditional_final_fp32_stage
#axiomsClean VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPCorrect.output_nonempty_rounding
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Reuse
