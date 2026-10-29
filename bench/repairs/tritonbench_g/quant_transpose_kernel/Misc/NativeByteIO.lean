import bench.repairs.tritonbench_g.quant_transpose_kernel.Misc.NativeReference
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.ByteIO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.ByteIO
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

theorem native (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (separated : KernelGrid.Separate p) (version : Bool) :
    (IO.native multiply rest p version).ObservesInteger (IO.byteView (IO.native multiply rest p version))
      (Reference.nativeTarget multiply rest p version) := by
  simpa only [IO.native,Reference.nativeTarget,native_formula] using
    observes (nativeAlgebra multiply rest) nativeRound p separated version

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.ByteIO
