import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.RealIO

namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
open scoped VeriTile.Triton.Masked2DKernelIO₂
noncomputable section

/-- Retained finite flat endpoint. This optional stronger allocation contract
requires disjoint regions; the public regional IO admits read-only aliases. -/
theorem dequantize_rowwise_flat_correct (p : Parameters) (positive : 0 < p.cols) :
    Spec.Real (RealIO.originalIO xRegion scaleRegion outRegion (1 / 127)
      (p.rows * p.cols) p.cols p.width ⊨ᵢ fun _ _ xs ys j => ys j * (xs j : ℝ) * (1 / 127)) :=
  RealIO.original_correct _ _ _ _ _ _ _ positive (lt_of_lt_of_le positive p.covers)

#axiomsClean dequantize_rowwise_flat_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise
