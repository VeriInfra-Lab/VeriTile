import bench.repairs.tritonbench_g.quant_transpose_kernel.IO
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.ReaderBefore
namespace VeriTile.Bench.Repairs.QuantTransposeKernel.LogicalReader
open VeriTile.Triton FP.Structural KernelModels
noncomputable section

def formula (p : KernelSource.Parameters) (xs : LogicalView.Inputs p)
    (column : Fin p.N) (row : Fin p.M) : Int :=
  ((BitVec.ofInt 8 (min (2^63-1) (max (-(2^63))
    (VeriTile.Bench.Repairs.RowwiseQuantizationTriton.roundEven
      (127*(xs.values row column*xs.inverse)))))).toNat : Int)

theorem judgment_iff (rest : Algebra ℝ) (p : KernelSource.Parameters) (version : Bool) :
    (LogicalView.transposeIO rest p version).Implements (formula p) ↔
    (ReaderBefore.kernelIO rest p version).ObservesInteger
      (IO.byteView (ReaderBefore.kernelIO rest p version)) (ReaderBefore.target rest p version) := Iff.rfl
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.LogicalReader
