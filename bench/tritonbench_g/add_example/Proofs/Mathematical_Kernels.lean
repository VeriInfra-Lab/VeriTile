import bench.tritonbench_g.add_example.Kernels
import VeriTile.Triton

/-! The Real proof uses the algorithm projection of the repair source.
No independent executable implementation is maintained here. -/
namespace VeriTile.Bench.TritonBenchG.AddExample
open VeriTile.Triton
noncomputable section

def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.AddExample.Contiguous.add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end
end VeriTile.Bench.TritonBenchG.AddExample
