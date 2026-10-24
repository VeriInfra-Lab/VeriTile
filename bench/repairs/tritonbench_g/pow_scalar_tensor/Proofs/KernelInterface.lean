import bench.repairs.tritonbench_g.pow_scalar_tensor.IO

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelInterface
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer

theorem view (v : SignedView.View) : (buffer v).view v.stride=v := rfl
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.KernelInterface
