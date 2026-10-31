import bench.tritonbench_g.iv_dependent_matmul.Sources.Selected
import bench.tritonbench_g.iv_dependent_matmul.Contract

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape
open Lean VeriTile.Triton
open TritonBenchG.IvDependentMatmul FPValue
namespace Kernel

/-! Complete ordinary quotations for each retained pointer schedule and graph.
The source graph has Dot+Add; the compiled graph has the existing live carry
operand. These are preserved program views, not a numerical rewrite identity.
All signed64/pointer correspondence remains proved at its original boundary. -/

def quotation (s : Kernels.Schedule) (g : Graph) (p : Parameters) : ComputeKernel :=
  (match s, g with
  | .preLoad, .source => preLoadSource
  | .postLoad, .source => postLoadSource
  | .mixed, .source => mixedSource
  | .two, .source => twoSource
  | .three, .source => threeSource
  | .preLoad, .accumulator => preLoadNative
  | .postLoad, .accumulator => postLoadNative
  | .mixed, .accumulator => mixedNative
  | .two, .accumulator => twoNative
  | .three, .accumulator => threeNative)
    .fp32 p.input p.weight p.output p.rows p.columns p.inner p.inputRowStride p.inputInnerStride
    p.weightInnerStride p.weightColumnStride p.outputRowStride p.outputColumnStride
    p.blockRows p.blockColumns p.blockInner

def source (r : Whole.Request) := quotation r.schedule .source r.parameters
def compiled (r : Whole.Request) := quotation r.schedule .accumulator r.parameters

end Kernel
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape
