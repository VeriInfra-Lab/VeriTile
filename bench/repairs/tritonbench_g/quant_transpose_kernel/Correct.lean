import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.QuantTransposeKernel.Correct
open VeriTile.Triton FP.Structural KernelModels
open scoped VeriTile.Triton.LogicalIO
noncomputable section
/-- Unsigned byte bits of nearest-even signed64-clamped quantization, at the
transposed logical [column,row] coordinate. This is an integer observation. -/
def target (p : KernelSource.Parameters) (xs : LogicalView.Inputs p)
    (column : Fin p.N) (row : Fin p.M) : Int :=
  ((BitVec.ofInt 8 (min (2^63-1) (max (-(2^63))
    (VeriTile.Bench.Repairs.RowwiseQuantizationTriton.roundEven
      (127*(xs.values row column*xs.inverse)))))).toNat : Int)

specification kernel_real (rest : Algebra ℝ) (p : KernelSource.Parameters)
    (separated : KernelGrid.Separate p) (version : Bool) :
    Spec.Real (LogicalView.transposeIO rest p version ⊨ target p) :=
  (LogicalReader.judgment_iff rest p version).mpr (ReaderBefore.kernel_real rest p separated version)
#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.Correct
