import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.CheckedCorrect
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.CheckedFPEquiv
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Kernel

/-! Complete Tensor applications of the raw full-grid kernel results. -/
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.TensorBridge
open VeriTile Triton FP.Structural Source Whole Kernel
noncomputable section

/-- Kernel-only correctness for full, noncausal attention with block scales:
sum_j 2^score(n,j) V[j,d] / sum_j 2^score(n,j), where score is the 96-term
QK product times query/key block scales. The wrapper supplies the checked
metadata and ordinary pre-kernel output; all logical Tensor coordinates and
the complete exterior region frame follow from quoted-source execution.
Real erases rounding and does not claim native-to-Real accuracy. -/
theorem attention_kernel_correct (p : Parameters) (args : Checked.Arguments p)
    (storage : TensorStorage p) (ordinary : Checked.OrdinaryPlacement p storage)
    (profile : Profile) (_contract : InputContract p args ordinary.base profile)
    (out : Descriptor) (output : OutputBinding p args out)
    (initial : State Real.R) (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        final.mem out.region (tensorOffset out b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠out.region ∨ ¬Footprint p off → final.mem r off=initial.mem r off)) := by
  simpa only [Spec.Real,output.region,output_offset p args out output] using
    KernelRealResult.attention_kernel_real p (Whole.separated p storage) storage.outputInjective initial μ reads work legal

/-- Exact native interpretation of the shared quotation at the kernel boundary.
The selected scalar and fixed K8 TF32/K16 FP16 instruction semantics are model
parameters. The result covers raw words, complete Tensor coordinates, FP16
output bytes and the exterior frame; no primitive law is an experiment TODO. -/
theorem attention_kernel_native (instructions : Native.Instructions)
    (p : Parameters) (args : Checked.Arguments p) (storage : TensorStorage p)
    (ordinary : Checked.OrdinaryPlacement p storage) (profile : Profile)
    (contract : InputContract p args ordinary.base profile)
    (out : Descriptor) (output : OutputBinding p args out)
    (initial : State Native.Word) (typed : Native.RawInputs p initial)
    (flat : Fin (Checked.admittedPlacement p storage ordinary contract.pointers).arena → BitVec 8)
    (binding : Native.WordBinding p storage
      (Checked.admittedPlacement p storage ordinary contract.pointers) flat initial)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ final,runPrograms (fun _=>Native.algebra) (fun _=>Native.dotModel) p (grid p) work
        (Native.injectState initial)=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        (Native.projectState instructions p initial final).mem out.region (tensorOffset out b h n d)=
          .mk .fp16 (NativeReference.target instructions p initial b h n d hd)) ∧
      (∀ r off,r≠out.region ∨ ¬Footprint p off →
        (Native.projectState instructions p initial final).mem r off=initial.mem r off) ∧
      NativeReference.ByteResult instructions p storage
        (Checked.admittedPlacement p storage ordinary contract.pointers) initial flat
        (flattenBytes p storage (Checked.admittedPlacement p storage ordinary contract.pointers)
          (fun _=>Native.encode16) flat (Native.projectState instructions p initial final)) ∧
      NativeReference.SourceReadback instructions p storage
        (Checked.admittedPlacement p storage ordinary contract.pointers) initial flat := by
  obtain ⟨final,run,values⟩:=RawKernel.native_result instructions p (Whole.separated p storage)
    storage.outputInjective initial work legal
  have bytes := Native.finite_result instructions p storage
    (Checked.admittedPlacement p storage ordinary contract.pointers) flat initial _ binding values
  refine ⟨final,run,?_,?_,⟨?_,bytes.2⟩,
    NativeReference.source_readback instructions p storage _ initial typed flat binding⟩
  · intro b hb h hh n hn d hd
    rw [output.region,output_offset p args out output,values.1 b hb h hh n hn d hd,
      NativeReference.target_binding instructions p initial b h n d hh hn hd]
  · simpa only [output.region] using values.2
  · intro b hb h hh n hn d hd k hk i address
    rw [bytes.1 b hb h hh n hn d hd k hk i address,
      NativeReference.target_binding instructions p initial b h n d hh hn hd]

#axiomsClean attention_kernel_correct
#axiomsClean attention_kernel_native
#axiomsClean NativeReference.source_readback
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.TensorBridge
