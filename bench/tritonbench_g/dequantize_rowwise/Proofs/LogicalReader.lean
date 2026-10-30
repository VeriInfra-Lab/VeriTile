import bench.tritonbench_g.dequantize_rowwise.IO
import bench.tritonbench_g.dequantize_rowwise.Proofs.ReaderReal

namespace VeriTile.Bench.Repairs.DequantizeRowwise.ReaderProof
open VeriTile.Triton
noncomputable section

/-- Equality of the complete masked IO judgments, including the scale access. -/
theorem logical_iff (p : Parameters) :
    (logicalIO p).Implements (fun xs j => xs.scale * (xs.weights j : ℝ) / 127) ↔
      (dequantizeIO p).Implements (fun weights scale j => scale 0 * (weights j : ℝ) / 127) :=
  Iff.rfl

theorem logical_correct (p : Parameters) :
    (logicalIO p).Implements (fun xs j => xs.scale * (xs.weights j : ℝ) / 127) :=
  (logical_iff p).mpr (dequantize_rowwise_correct p)

end
end VeriTile.Bench.Repairs.DequantizeRowwise.ReaderProof
