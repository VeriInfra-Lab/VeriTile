import bench.repairs.tritonbench_g.quant_transpose_kernel.Misc.NativeReference
import bench.repairs.tritonbench_g.quant_transpose_kernel.Misc.NativeByteIO

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
open VeriTile.Triton FP.Structural KernelModels
noncomputable section

abbrev kernelIO := IO.native
abbrev target := Reference.nativeTarget

/-- The actual candidate only commutes one integer input-row address product. -/
def commuted (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : KernelSource.Parameters) :
    IntegerProgramIO.Variant (kernelIO multiply rest p Bool.false) :=
  IO.commuted (nativeAlgebra multiply rest) nativeRound p

/-- The independent native byte formula and complete source-pair equivalence.
All FP32 words are admitted. Both source-ordered multiply calls and the pinned
sm80 signed64/FTZ conversion are primitive model data, with no numerical rewrite
assumptions or deferred primitive-conformance experiments. -/
theorem kernel_native_observed_equiv (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : KernelSource.Parameters) (separated : KernelGrid.Separate p) :
    (kernelIO multiply rest p Bool.false).ObservesInteger
      (IO.byteView (kernelIO multiply rest p Bool.false)) (target multiply rest p Bool.false) ∧
    (kernelIO multiply rest p Bool.false).IntegerObservedEquivalent (commuted multiply rest p)
      (IO.byteView (kernelIO multiply rest p Bool.false)) :=
  ⟨ByteIO.native multiply rest p separated Bool.false,
    ByteIO.equivalent (nativeAlgebra multiply rest) nativeRound p separated⟩

alias native_nan_byte := KernelModels.nan_byte
#axiomsClean kernel_native_observed_equiv
#axiomsClean native_nan_byte
#specNonCircular nativeByte avoiding [KernelSource.kernel,KernelExecution.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
