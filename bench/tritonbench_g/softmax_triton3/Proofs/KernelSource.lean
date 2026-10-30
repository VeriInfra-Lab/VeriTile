import bench.tritonbench_g.softmax_triton3.Kernels
import bench.tritonbench_g.softmax_triton3.Proofs.Mathematical_Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.SoftmaxTriton3
open VeriTile.Triton

/-- The selected FP32 storage declaration makes the readable source explicit;
the legacy source already inferred precisely this FP32 computation. -/
theorem source_legacy (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) :
    softmax_kernel out inp mask stride n B hasMask =
      VeriTile.Bench.TritonBenchG.SoftmaxTriton3.softmax_kernel out inp mask stride n B hasMask := by
  cases hasMask <;> rfl

#axiomsClean source_legacy
end VeriTile.Bench.Repairs.SoftmaxTriton3
