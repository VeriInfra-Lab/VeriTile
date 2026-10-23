import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Bytes
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.Rotary
namespace VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Bench.Repairs.RmsMatmulRbe
noncomputable section
abbrev rms_matmul_rbe := @value
abbrev rms_matmul_rbe_wrapper := @hostCode
abbrev rbe_triton := @VeriTile.Bench.Repairs.RbeTritonTransform.Native.entry
end
end VeriTile.Bench.Repairs.RmsRbeMatmul
