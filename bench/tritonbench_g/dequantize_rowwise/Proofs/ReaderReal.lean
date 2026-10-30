import bench.tritonbench_g.dequantize_rowwise.Proofs.KernelCorrectness
import bench.tritonbench_g.dequantize_rowwise.IO

namespace VeriTile.Bench.Repairs.DequantizeRowwise.ReaderProof
open VeriTile VeriTile.Triton
noncomputable section

theorem dequantize_rowwise_correct (p : Parameters) :
    (dequantizeIO p).Implements (fun weights scale j => scale 0 * (weights j : ℝ) / 127) :=by
  refine MixedMaskedKernelIO₂.Implements.intro _ ?_ ?_
  · intro bounds s weights scale output
    exact KernelCorrectness.traceSafe p bounds s weights (scale 0 trivial) output
  · intro s weights scale weightsRead scaleRead
    exact KernelCorrectness.inputs_correct p s weights (scale 0)
      weightsRead (scaleRead 0 trivial)

end
end VeriTile.Bench.Repairs.DequantizeRowwise.ReaderProof
