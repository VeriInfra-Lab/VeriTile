import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeRowwise.ReadableSource
open VeriTile.Bench.TritonBenchG.DequantizeRowwise

/-- Complete functions, including source IO, all parameters and every statement. -/
theorem original_preserved : source=TypedSource.original := rfl
theorem candidate_preserved : indexCommutedSource=TypedSource.indexCommuted := rfl
theorem kernel_preserved (p : Parameters) :
    kernel p=TypedSource.original xRegion scaleRegion outRegion (1/127) (p.rows*p.cols) p.cols p.width := rfl
theorem candidate_kernel_preserved (p : Parameters) :
    indexCommutedKernel p=TypedSource.indexCommuted xRegion scaleRegion outRegion (1/127)
      (p.rows*p.cols) p.cols p.width := rfl

#axiomsClean original_preserved
#axiomsClean candidate_preserved
end VeriTile.Bench.Repairs.DequantizeRowwise.ReadableSource
