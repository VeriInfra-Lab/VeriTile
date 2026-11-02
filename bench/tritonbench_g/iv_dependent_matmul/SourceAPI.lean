import bench.tritonbench_g.iv_dependent_matmul.Kernels
import bench.tritonbench_g.iv_dependent_matmul.Contract

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
open TritonBenchG.IvDependentMatmul FPValue

def source (r : Whole.Request) := quotation r.schedule .source r.parameters
def compiled (r : Whole.Request) := quotation r.schedule .accumulator r.parameters

end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
