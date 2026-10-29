import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
open VeriTile.Triton FP.Structural Physical Host
noncomputable section

/-- Every FP32 bit pattern is copied unchanged, or literal positive zero is
stored. There is no floating arithmetic and no numerical assumption. -/
specification embedding_wrapper_fp32 (rest : Algebra Word) (c : Request) (h : InputContract c)
    (storage : Native.TensorStorage c) (before : Memory) (outputObject : Nat)
    (incoming : VeriTile.Torch.Device) :
    ∃ t,typedRun rest 0 id c before=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real (reference 0 id c before r d) ∧
        (run c before outputObject incoming).memory
          (address c.output (r*c.hidden+d))=reference 0 id c before r d) ∧
      (run c before outputObject incoming).value=none ∧
      (run c before outputObject incoming).outputObject=outputObject ∧
      (run c before outputObject incoming).outputMetadata=c.output ∧
      (run c before outputObject incoming).restoredDevice=incoming ∧
      (∀ byte,Outside c byte → (run c before outputObject incoming).memory byte=before byte) :=
  whole rest 0 id rfl c h storage before outputObject incoming

/-- Complete typed source execution and readback through the caller's logical
Tensor values. Ordinary Tensor laws derive storage bounds from the same checked
metadata; no source-output equation is assumed. -/
specification embedding_tensor_fp32 (rest : Algebra Word) (c : Request) (h : InputContract c)
    (objects : TensorBinding.Objects c) (before : Memory) (incoming : VeriTile.Torch.Device) :
    ∃ t,typedRun rest 0 id c before=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real (TensorBinding.reference c objects before r d) ∧
        TensorBinding.outputValue c objects
          (Host.run c before objects.output.object incoming).memory r d=
            TensorBinding.reference c objects before r d) ∧
      (Host.run c before objects.output.object incoming).value=none ∧
      (Host.run c before objects.output.object incoming).outputObject=objects.output.object ∧
      (Host.run c before objects.output.object incoming).outputMetadata=c.output ∧
      (∀ byte,OutsideByte c byte →
        readByte (Host.run c before objects.output.object incoming).memory byte=readByte before byte) := by
  obtain ⟨t,exec,values,_⟩ := typed_correct rest 0 id c before
  obtain ⟨tensor,_,_⟩ := TensorBinding.returned_tensor c h objects before incoming
  refine ⟨t,exec,?_,rfl,rfl,rfl,?_⟩
  · intro r hr d hd
    refine ⟨?_,tensor r hr d hd⟩
    have value := values r hr d hd
    change t.mem "out" (r*c.hidden+d)=.mk .real (desired c before r d) at value
    rw [TensorBinding.desired_logical c h objects before r d hr hd] at value
    exact value
  · exact returned_byte_frame c h (TensorBinding.storage c h objects) before objects.output.object incoming

#axiomsClean embedding_tensor_fp32
#specNonCircular TensorBinding.reference avoiding [typedRun,Host.run,Native.launch,Physical.launch]

export Native (accesses signed_offsets rows_exact normalized_endpoints all_lane_offsets initial_integer_casts access_alignment)
export Host (returned_byte_frame)
#specNonCircular reference avoiding [typedRun,Host.run,Native.launch,Physical.launch]
#axiomsClean embedding_wrapper_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
