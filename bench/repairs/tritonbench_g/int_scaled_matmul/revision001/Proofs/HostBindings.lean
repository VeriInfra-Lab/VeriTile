import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Host
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Kernels
open VeriTile.Triton
noncomputable section
/-- Both public launches select the repaired typed source, with host strides. -/
def unscaled (x : Host.Arguments) := BlockWord.source (Host.blockParameters x)
def scaled (x : Host.Arguments) (scale : Host.Tensor) := ScaledWord.source (Host.scaledParameters x scale)
theorem unscaled_binding (x : Host.Arguments) : unscaled x=BlockWord.source (Host.blockParameters x) := rfl
theorem scaled_binding (x : Host.Arguments) (scale : Host.Tensor) :
    scaled x scale=ScaledWord.source (Host.scaledParameters x scale) := rfl
#axiomsClean unscaled_binding
#axiomsClean scaled_binding
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Kernels
