import VeriTile.Triton
import VeriTile.Triton.DSL.AlgorithmView
import bench.tritonbench_g.vector_addition.Kernels

namespace VeriTile.Bench.TritonBenchG.VectorAddition
open VeriTile.Triton

/-- Mathematical interpretation of the one current source. Its existing Algorithm
projection is used only for mathematical interpretation; the tagged source remains authoritative. -/
def add_kernel (x_ptr y_ptr output_ptr : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  source_algorithm_view (VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel
    x_ptr y_ptr output_ptr n_elements BLOCK_SIZE)

end VeriTile.Bench.TritonBenchG.VectorAddition
