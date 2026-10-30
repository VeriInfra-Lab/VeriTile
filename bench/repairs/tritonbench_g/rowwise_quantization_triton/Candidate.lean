import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Candidate
open VeriTile.Triton

/-- Store the completed FP16 maximum before quantization, shortening its live
range across the divide, scale, llrint and INT8 store. Numerical operations,
source conversions and both output channels are otherwise unchanged. -/
abbrev source := RowwiseQuantizationTriton.earlyMaximumSource

end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Candidate
