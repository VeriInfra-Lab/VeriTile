import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.AcceptedReal
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Contract
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16AcceptedReal
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Contract

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned Source Whole Full Real
noncomputable section
specification aligned_real (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State R) (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebra) (fun _=>model) p abi before work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        result.state.mem p.output (logicalAddress p b h n d)=.mk .fp16 (some (expected p μ b h n d))) ∧
      (∀ r off,OutsideLogical p r off → result.state.mem r off=before.mem r off)) :=
  Real.correct p (Contract.guard p observed admitted) storage abi before μ h work legal

#axiomsClean aligned_real
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Correct
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.BF16 Source Whole Full Real
noncomputable section
specification aligned_real (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State R) (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebra) (fun _=>model) p abi before work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        result.state.mem p.output (logicalAddress p b h n d)=.mk .bf16 (some (expected p μ b h n d))) ∧
      (∀ r off,OutsideLogical p r off → result.state.mem r off=before.mem r off)) :=
  Real.correct p (Contract.guard p observed admitted) storage abi before μ h work legal

#axiomsClean aligned_real
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Correct
