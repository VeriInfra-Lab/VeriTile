import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.EmptyReturn
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.AcceptedReal
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Contract
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Correct
open VeriTile Triton FP.Structural Source Whole Real
noncomputable section
abbrev target := Real.expected
specification causal_real (p : Parameters) (observed : Contract.Observations p) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State R) (alloc : Allocation p abi before)
    (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi before alloc work=some result ∧
      result.tensor=alloc.output ∧
      result.tensor.shape=[p.batches,p.heads,p.length,128] ∧
      result.tensor.dtype=.fp16 ∧ result.tensor.device=abi.query.device ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out → r≠alloc.unusedM.region → result.state.mem r off=before.mem r off)) := by
  obtain ⟨result,he,ht,hv,hf⟩ := Real.correct p (Contract.guard p observed admitted) storage abi before alloc μ h work legal
  refine ⟨result,he,ht,?_,?_,?_,hv,hf⟩
  · rw [ht,alloc.outputShape,abi.shapes.1]
  · rw [ht];exact alloc.outputType
  · rw [ht];exact alloc.outputDevice

#specNonCircular target avoiding [Whole.invoke,FP.Structural.exec]
#axiomsClean causal_real
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Correct
