import bench.tritonbench_g.fifth_order_sph_harmonics.Kernels
import VeriTile.Triton
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  input : RegionName
  output : RegionName
  blockSize : Nat
  coordNumel : Nat
  outputNumel : Nat
  columnOffset : Nat
  outputStride : Nat

/-- Full original eleven-channel forward graph. The source FP32 profile makes
constexpr coefficient types explicit; Y07 retains the original subtraction,
and unary negation is retained instead of real-algebra reassociation. -/
def source (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) : ComputeKernel := VeriTile.Bench.Repairs.FifthOrderSphHarmonics.forwardPastedKernel coord_ptr output_ptr block_size coord_numel output_numel col_offset output_stride
def kernel (p : Parameters) := source p.input p.output p.blockSize p.coordNumel p.outputNumel p.columnOffset p.outputStride
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
