import bench.repairs.tritonbench_g.attn_fwd_causal.Misc.RawCorrect
import bench.repairs.tritonbench_g.attn_fwd_causal.Misc.RawNative
import bench.repairs.tritonbench_g.attn_fwd_causal.Misc.HistoricalCorrect
import bench.repairs.tritonbench_g.attn_fwd_causal.Misc.HistoricalFPEquiv

/-! Complete checked Tensor applications of the raw kernel results. -/
namespace VeriTile.Bench.Repairs.AttnFwdCausal.TensorBridge
open VeriTile Triton FP.Structural Source Whole Kernel
noncomputable section

/-- Complete kernel-only causal attention, with all 128 output features and
an independent base-2 weighted-sum formula. The wrapper supplies assertions;
Real erases rounding and does not claim a native accuracy bound. -/
theorem causal_kernel_correct (p : Parameters) (observed : Contract.Observations p)
    (_admitted : InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (out : Descriptor) (output : OutputBinding p abi out)
    (initial : State Real.R) (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        final.mem out.region (tensorOffset out b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠out.region ∨ ¬Footprint p off → final.mem r off=initial.mem r off)) := by
  obtain ⟨final,run,values,frame⟩:=RawCorrect.kernel_real p storage.separated initial μ reads work legal
  refine ⟨final,run,?_,?_⟩
  · intro b hb h hh n hn d hd
    rw [output.region,output_offset p abi out output b h n d hb hh hn]
    exact values b hb h hh n hn d hd
  · simpa only [output.region] using frame

#axiomsClean causal_kernel_correct
#specNonCircular Real.expected avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]

/-- Exact native interpretation of the shared quotation at the kernel boundary.
The selected scalar and fixed K8 TF32/K16 FP16 instruction semantics are model
parameters. The result covers raw words, complete Tensor coordinates, FP16
output bytes and the exterior frame; no primitive law is an experiment TODO. -/
theorem causal_kernel_native (instructions : Native.Instructions)
    (p : Parameters) (observed : Contract.Observations p) (admitted : InputContract p observed)
    (storage : TensorStorage p) (abi : InputABI p) (api : Contract.StorageAPI p storage observed)
    (out : Descriptor) (output : OutputBinding p abi out)
    (initial : State Native.Word) (typed : Native.RawInputs p initial)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8)
    (binding : Native.WordBinding p storage (Contract.placement p storage observed api admitted) flat initial)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        (Native.projectState instructions p initial final).mem out.region (tensorOffset out b h n d)=
          .mk .fp16 (NativeReference.target instructions p initial b h n d hd)) ∧
      (∀ r off,r≠out.region ∨ ¬Footprint p off →
        (Native.projectState instructions p initial final).mem r off=initial.mem r off) ∧
      NativeReference.ByteResult instructions p storage
        (Contract.placement p storage observed api admitted) initial flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted)
          (fun _=>Native.encode16) flat (Native.projectState instructions p initial final)) ∧
      NativeReference.SourceReadback instructions p storage
        (Contract.placement p storage observed api admitted) initial flat := by
  obtain ⟨final,run,values,frame⟩ := RawNative.kernel_native instructions p storage.separated initial work legal
  have words : Native.WordResult instructions p initial (Native.projectState instructions p initial final) := by
    refine ⟨?_,frame⟩
    intro b hb h hh n hn d hd
    rw [values b hb h hh n hn d hd,
      NativeReference.target_binding instructions p initial b h n d hh hn hd]
  have bytes := Native.finite_result instructions p storage
    (Contract.placement p storage observed api admitted) flat initial
    (Native.projectState instructions p initial final) binding words
  refine ⟨final,run,?_,?_,⟨?_,bytes.2⟩,
    NativeReference.source_readback instructions p storage _ initial typed flat binding⟩
  · intro b hb h hh n hn d hd
    rw [output.region,output_offset p abi out output b h n d hb hh hn]
    exact values b hb h hh n hn d hd
  · simpa only [output.region] using frame
  · intro b hb h hh n hn d hd k hk i address
    rw [bytes.1 b hb h hh n hn d hd k hk i address,
      NativeReference.target_binding instructions p initial b h n d hh hn hd]

#axiomsClean causal_kernel_native
#axiomsClean Native.input_from_current_bytes
#axiomsClean NativeReference.source_load_word
#axiomsClean NativeReference.source_readback
#specNonCircular NativeReference.target avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]

end
end VeriTile.Bench.Repairs.AttnFwdCausal.TensorBridge
