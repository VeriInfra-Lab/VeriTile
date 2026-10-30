import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Correct
open VeriTile.Triton FP.Structural Physical Host
noncomputable section

/-- Mathematical gather over initial input values, with zero for every ID
outside the original exclusive interval. The actual typed source executes
in the Real interpretation and the original output object receives that result. -/
specification embedding_wrapper_real (rest : Algebra ℝ) (interpret : Word → ℝ)
    (positive_zero : interpret 0=0) (c : Request) (h : InputContract c)
    (storage : Native.TensorStorage c) (before : Memory) (outputObject : Nat)
    (incoming : VeriTile.Torch.Device) :
    ∃ t,typedRun rest 0 interpret c before=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real (reference 0 interpret c before r d) ∧
        interpret ((run c before outputObject incoming).memory
          (address c.output (r*c.hidden+d)))=reference 0 interpret c before r d) ∧
      (run c before outputObject incoming).value=none ∧
      (run c before outputObject incoming).outputObject=outputObject ∧
      (run c before outputObject incoming).outputMetadata=c.output ∧
      (run c before outputObject incoming).restoredDevice=incoming ∧
      (∀ byte,Outside c byte → (run c before outputObject incoming).memory byte=before byte) :=
  whole rest 0 interpret positive_zero c h storage before outputObject incoming

specification embedding_tensor_real (rest : Algebra ℝ) (interpret : Word → ℝ)
    (positive_zero : interpret 0=0) (c : Request) (h : InputContract c)
    (objects : TensorBinding.Objects c) (before : Memory) (incoming : VeriTile.Torch.Device) :
    ∃ t,typedRun rest 0 interpret c before=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real
          (TensorBinding.interpretedReference 0 interpret c objects before r d) ∧
        interpret (TensorBinding.outputValue c objects
          (Host.run c before objects.output.object incoming).memory r d)=
            TensorBinding.interpretedReference 0 interpret c objects before r d) ∧
      (Host.run c before objects.output.object incoming).value=none ∧
      (Host.run c before objects.output.object incoming).outputObject=objects.output.object ∧
      (∀ byte,OutsideByte c byte →
        readByte (Host.run c before objects.output.object incoming).memory byte=readByte before byte) := by
  obtain ⟨t,exec,values,_⟩ := typed_correct rest 0 interpret c before
  obtain ⟨tensor,_,_⟩ := TensorBinding.returned_tensor c h objects before incoming
  refine ⟨t,exec,?_,rfl,rfl,?_⟩
  · intro r hr d hd
    have formula : Host.reference 0 interpret c before r d=
        TensorBinding.interpretedReference 0 interpret c objects before r d := by
      rw [←interpret_desired 0 interpret positive_zero,
        TensorBinding.desired_logical c h objects before r d hr hd,
        TensorBinding.interpret_reference 0 interpret positive_zero]
    exact ⟨(values r hr d hd).trans (congrArg (Cell.mk .real) formula),
      by rw [tensor r hr d hd,TensorBinding.interpret_reference 0 interpret positive_zero]⟩
  · exact returned_byte_frame c h (TensorBinding.storage c h objects) before objects.output.object incoming

#axiomsClean embedding_tensor_real
#specNonCircular TensorBinding.interpretedReference avoiding [typedRun,Host.run,Native.launch,Physical.launch]

export VeriTile.Bench.Repairs.EmbeddingTritonKernel (checks_iff argument_checks_iff)
#specNonCircular reference avoiding [typedRun,Host.run,Native.launch,Physical.launch]
#axiomsClean embedding_wrapper_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Correct
