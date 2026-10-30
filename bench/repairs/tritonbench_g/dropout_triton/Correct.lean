import bench.repairs.tritonbench_g.dropout_triton.Proofs.RealEndpoints
import bench.repairs.tritonbench_g.dropout_triton.IO
import VeriTile.Meta.StatementAudit

/-! Public mixed-channel IO for the exact typed dropout sources. The Real
formula uses total division; the conventional partial formula and its declared
kept-coordinate domain are retained in Proofs/RealEndpoints. Input views may
share physical bytes. No flat-allocation separation or new input guard is added.
-/
namespace VeriTile.Bench.Repairs.DropoutTriton.Correct
open VeriTile Triton
open scoped VeriTile.Triton.MixedMaskedKernelIO₂
noncomputable section

/-- FP32 data, signed-int32 keep mask and FP32 output share one masked tile.
The source keeps these precisions; Real IO changes only their mathematical
interpretation. There are no private scratch writes. -/
def dropoutIO (n : Nat) (p : ℝ) : MixedMaskedKernelIO₂ := IO.dropoutIO n p

specification dropout_kernel_correctness (n : Nat) (p : ℝ) :
    Spec.Real (dropoutIO n p ⊨ fun (xs : Fin blockSize → ℝ) (keep : Fin blockSize → Int) j =>
      (if keep j≠0 then xs j/(1-p) else 0 : ℝ)) := by
  apply MixedMaskedKernelIO₂.Implements.intro
  · exact RealEndpoints.masked_safe n p
  · exact RealEndpoints.masked_run n p

#stmtSurfaceSubset dropout_kernel_correctness ⊆
  [Spec.Real, blockSize, dropoutIO, MixedMaskedKernelIO₂.Implements,
   MixedMaskedKernelIO₂.B1, MixedMaskedKernelIO₂.B2, MixedMaskedKernelIO₂.Bout,
   MixedMaskedKernelIO₂.type1, MixedMaskedKernelIO₂.type2, MixedMaskedKernelIO₂.outType,
   ChanTy.carrier]
#specNonCircular expected avoiding [kernel, indexCommutedKernel]
#specNonCircular mathematicalValue avoiding [kernel, indexCommutedKernel]
#axiomsClean dropout_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Correct
