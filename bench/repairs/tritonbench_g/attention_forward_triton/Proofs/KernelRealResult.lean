import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.KernelRealResult
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Complete noncausal attention on raw region memory, with arbitrary legal
full-grid order, the independent Real target, and every exterior cell. -/
theorem attention_kernel_real (p : Parameters) (sep : Separated p)
    (injective : RawKernel.OutputInjective p) (initial : State Real.R)
    (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off)) :=
  RawKernel.real_result p sep injective initial μ reads work legal

#axiomsClean attention_kernel_real
#specNonCircular Real.expected avoiding [runPrograms,FP.Structural.exec,accumulated]
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.KernelRealResult
