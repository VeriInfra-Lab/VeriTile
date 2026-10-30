import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RawCorrect
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- Complete shared-source grid execution and the independent causal formula
at every raw output address, with the complete exterior regional frame. -/
theorem kernel_real (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (initial : State Real.R) (μ : RegionName → Nat → ℝ) (reads : Real.Reads p initial μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) p (grid p) work initial=some final ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        final.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (Real.expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out ∨ ¬Footprint p off → final.mem r off=initial.mem r off)) :=
  Kernel.real_result p (RawKernel.storage p apart) initial μ reads work legal

#axiomsClean kernel_real
#specNonCircular Real.expected avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.RawCorrect
