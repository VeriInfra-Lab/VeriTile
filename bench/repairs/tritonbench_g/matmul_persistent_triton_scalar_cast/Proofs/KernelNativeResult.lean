import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.RawKernel
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Native

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.KernelNativeResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Post)
noncomputable section

/-- Successful complete native source grid, all independent output words and
full typed frame, with the unchanged scalar and fixed-ISA instruction ports.
No Tensor object, API flags, wrapper result or admission is a premise. -/
theorem kernel_fp_equiv (p : Core.Parameters) (b : RawKernel.Buffers)
    (initial : State Word) (entry : Memory Word) (layout : RawKernel.Layout p b)
    (inputs : RawKernel.Inputs p b initial entry) (sms : 0<p.problem.SMS)
    (plan : DotLayout p.problem.profile.blockK) (native : Numerics) (reference : Native.Reference)
    (ports : LocalPorts native reference.numerics Native.allWords) :
    ∃ final,Core.executeGrid (Native.arithmetic p.problem.profile plan native) p initial=some final ∧
      Post "C" (Core.logicalAddress p)
        (fun x => Core.pack p.problem.profile
          (RawKernel.reference (Native.arithmetic p.problem.profile plan reference.numerics)
            p b entry x.1.val x.2.val)) (fun _ => True) initial final := by
  obtain ⟨final,ran,values,frame⟩ := Core.gridExecution
    (Native.arithmetic p.problem.profile plan native) p (b.descriptors p) initial entry
    (layout.descriptors p b) (RawKernel.strides p b) (inputs.descriptors p b initial entry)
    (Native.padding_zero p.problem.profile plan native reference ports) sms
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  unfold Core.referenceValue RawKernel.reference
  rw [Native.target_equivalent p.problem (b.descriptors p) entry plan native reference.numerics ports]

#axiomsClean kernel_fp_equiv
#specNonCircular LocalPorts avoiding [Core.executeGrid,Targets.result,Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.KernelNativeResult
