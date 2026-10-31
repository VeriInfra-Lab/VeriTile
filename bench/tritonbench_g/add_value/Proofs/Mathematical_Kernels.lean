import bench.tritonbench_g.add_value.Kernels
import VeriTile.Triton

/-! The Real proof uses the algorithm projection of the repair source.
No independent executable implementation is maintained here. -/
namespace VeriTile.Bench.TritonBenchG.AddValue
open VeriTile.Triton
noncomputable section

def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) :
    ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.AddValue.Contiguous.kernel x_ptr output_ptr N BLOCK_SIZE).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end
end VeriTile.Bench.TritonBenchG.AddValue
