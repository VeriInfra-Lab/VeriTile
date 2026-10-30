import bench.tritonbench_g.quant_transpose_kernel.IO
import bench.tritonbench_g.quant_transpose_kernel.Proofs.ByteIO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.ReaderBefore
open VeriTile.Triton FP.Structural KernelModels
noncomputable section

/-- Strided FP32 matrix, FP32 inverse scalar, transposed INT8 output and the
complete original grouped grid. Both retained integer-index versions use it. -/
abbrev kernelIO := IO.real
/-- Low eight bits are observed as an unsigned integer, after nearest-even
rounding and signed64 clamping. This is an integer observation, not a Real cast. -/
def target (rest : Algebra ℝ) (p : KernelSource.Parameters) (version : Bool) :=
  IO.formula (realAlgebra rest) realRound p version (fun x inverse =>
    ((BitVec.ofInt 8 (min (2^63-1) (max (-(2^63))
      (VeriTile.Bench.Repairs.RowwiseQuantizationTriton.roundEven (127*(x*inverse)))))).toNat : Int))


/-- Byte bits of nearest-even/clamped signed64 rounding of
127 * (input[row,column] * inverse), at output[column,row].
The shared judgment includes actual source safety, success and exact frame. -/
theorem kernel_real (rest : Algebra ℝ) (p : KernelSource.Parameters)
    (separated : KernelGrid.Separate p) (version : Bool) :
    Spec.Real ((kernelIO rest p version).ObservesInteger
      (IO.byteView (kernelIO rest p version)) (target rest p version)) :=
  ByteIO.real rest p separated version

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.ReaderBefore
