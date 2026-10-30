import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.KernelNativeResult
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Exact full-grid source execution and independent native attention graph on
raw word memory. Scalar and fixed K8 TF32/K16 FP16 ports are unchanged. -/
theorem attention_kernel_native_raw (instructions : Native.Instructions)
    (p : Parameters) (sep : Separated p) (injective : RawKernel.OutputInjective p)
    (initial : State Native.Word) (work : List Program)
    (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        (Native.projectState instructions p initial final).mem p.Out (logicalAddress p b h n d)=
          .mk .fp16 (NativeReference.target instructions p initial b h n d hd)) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off →
        (Native.projectState instructions p initial final).mem r off=initial.mem r off) := by
  obtain ⟨final,run,values,frame⟩ := RawKernel.native_result instructions p sep injective initial work legal
  refine ⟨final,run,?_,frame⟩
  intro b hb h hh n hn d hd
  rw [values b hb h hh n hn d hd,NativeReference.target_binding instructions p initial b h n d hh hn hd]

#axiomsClean attention_kernel_native_raw
#specNonCircular NativeReference.target avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.KernelNativeResult
