import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelWhole

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.RawCorrect
open VeriTile.Triton FP.Structural KernelModels
noncomputable section

/-- Complete grouped Real kernel launches on raw arrays and typed input reads. -/
theorem kernel_real (rest : Algebra ℝ) (p : KernelSource.Parameters)
    (separated : KernelGrid.Separate p) (values : Nat → Nat → ℝ)
    (inverse : ℝ) (initial : State ℝ) (reads : KernelWhole.Reads p values inverse initial) :
    ∃ final,
      KernelGrid.run (realAlgebra rest) realRound p Bool.false
        (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) initial=some final ∧
      KernelGrid.run (realAlgebra rest) realRound p Bool.true
        (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) initial=some final ∧
      KernelWhole.Result p (fun r c => realByte (values r c) inverse) initial final := by
  obtain ⟨final,he,result⟩ := KernelWhole.real_correct rest p separated values inverse initial reads
  exact ⟨final,he,(KernelGrid.same_run _ _ _ _ _).symm.trans he,result⟩

alias normalized_nonzero := KernelModels.normalized
#axiomsClean kernel_real
#specNonCircular realByte avoiding [KernelSource.kernel,KernelExecution.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.RawCorrect
