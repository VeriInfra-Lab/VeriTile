import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Host
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.NativeStaging
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
open VeriTile.Triton FP.Structural
/-- Ordinary raw tensor storage binding; no graph, arithmetic or result law. -/
def InputWords (a : Host.Arguments) (s : State Expr) : Prop :=
  ∀ t∈[a.a,a.s], ∀ o<t.allocation-t.storageOffset,
    (t.dtype=.fp16 → ∃ w : BitVec 16, s.mem t.region o=.mk .fp16 (.bits16 w)) ∧
    (t.dtype=.fp32 → ∃ w : BitVec 32, s.mem t.region o=.mk .fp32 (.bits32 w))
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
