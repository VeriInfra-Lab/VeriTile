import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelWhole

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.RawNative
open VeriTile.Triton FP.Structural KernelModels
noncomputable section

/-- Both complete raw native launches have the same output bytes and frame. -/
theorem kernel_native_equiv (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : KernelSource.Parameters) (separated : KernelGrid.Separate p)
    (values : Nat → Nat → Word) (inverse : Word) (initial : State Word)
    (reads : KernelWhole.Reads p values inverse initial) :
    ∃ final,
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound p Bool.false
        (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) initial=some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound p Bool.true
        (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) initial=some final ∧
      KernelWhole.Result p (fun r c => nativeByte multiply (values r c) inverse) initial final := by
  obtain ⟨final,he,result⟩ := KernelWhole.native_correct multiply rest p separated values inverse initial reads
  exact ⟨final,he,(KernelGrid.same_run _ _ _ _ _).symm.trans he,result⟩

alias native_nan_byte := KernelModels.nan_byte
#axiomsClean kernel_native_equiv
#axiomsClean native_nan_byte
#specNonCircular nativeByte avoiding [KernelSource.kernel,KernelExecution.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.RawNative
