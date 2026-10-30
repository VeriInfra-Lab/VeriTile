import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.LogicalReader
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.KernelCorrectness
import bench.repairs.tritonbench_g.dequantize_rowwise.IO

/-! Selected INT8 weights, FP32 scale arithmetic and FP16 output. Real is the
independent mathematical interpretation; source precision remains in Kernels. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Actual source access safety, successful execution, scale·weight/127 at
all active outputs, and every exterior regional memory cell. -/
specification dequantize_rowwise_correct (p : Parameters) :
    Spec.Real (logicalIO p ⊨ fun xs j => xs.scale * (xs.weights j : ℝ) / 127) :=
  ReaderProof.logical_correct p

#specNonCircular KernelCorrectness.dequantized avoiding [kernel, source, indexCommutedKernel]
#axiomsClean dequantize_rowwise_correct
#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset dequantize_rowwise_correct ⊆
  [VeriTile.Bench.Repairs.DequantizeRowwise.LogicalInputs,
   VeriTile.Bench.Repairs.DequantizeRowwise.LogicalInputs.scale,
   VeriTile.Bench.Repairs.DequantizeRowwise.LogicalInputs.weights,
   VeriTile.Bench.Repairs.DequantizeRowwise.Parameters,
   VeriTile.Bench.Repairs.DequantizeRowwise.Parameters.width,
   VeriTile.Bench.Repairs.DequantizeRowwise.dequantizeIO,
   VeriTile.Bench.Repairs.DequantizeRowwise.logicalIO,
   VeriTile.Spec.Real,
   VeriTile.Triton.ChanTy.carrier,
   VeriTile.Triton.LogicalIO.Implements,
   VeriTile.Triton.MixedMaskedKernelIO₂.B1,
   VeriTile.Triton.MixedMaskedKernelIO₂.B2,
   VeriTile.Triton.MixedMaskedKernelIO₂.Implements,
   VeriTile.Triton.MixedMaskedKernelIO₂.type1,
   VeriTile.Triton.MixedMaskedKernelIO₂.type2]

end
end VeriTile.Bench.Repairs.DequantizeRowwise
