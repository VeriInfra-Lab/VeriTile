import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.RawKernel

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.KernelRealResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Post)
noncomputable section

/-- Complete persistent source grid with the independent K-sum and full typed
frame. Premises contain only raw allocations, source geometry and raw reads. -/
theorem kernel_correct (p : Core.Parameters) (b : RawKernel.Buffers)
    (initial : State ℝ) (entry : Memory ℝ) (layout : RawKernel.Layout p b)
    (inputs : RawKernel.Inputs p b initial entry) (sms : 0<p.problem.SMS) :
    ∃ final,Core.executeGrid Real.arithmetic p initial=some final ∧
      Post "C" (Core.logicalAddress p)
        (fun x => Core.pack p.problem.profile
          (RealMath.formula p.problem.K (RawKernel.realInputs p b entry) x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,ran,values,frame⟩ := Core.gridExecution Real.arithmetic p (b.descriptors p) initial entry
    (layout.descriptors p b) (RawKernel.strides p b) (inputs.descriptors p b initial entry) rfl sms
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  change Core.pack p.problem.profile (RawKernel.reference Real.arithmetic p b entry x.1.val x.2.val)=_
  rw [RawKernel.real_reference p b entry sms x.1.val x.2.val x.1.isLt x.2.isLt]

#axiomsClean kernel_correct
#specNonCircular RealMath.formula avoiding [Core.executeGrid,Source.run]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.KernelRealResult
