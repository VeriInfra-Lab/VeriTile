import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RealCorrect
import bench.repairs.tritonbench_g.attention_forward_triton.Contract
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
open VeriTile Triton FP.Structural Source Whole Real
noncomputable section
theorem checked_real (p : Parameters) (strides : strideChecks p=Bool.true) (launch : launchChecks p=Bool.true) (storage : TensorStorage p)
    (ordinary : OrdinaryPlacement p storage) (_pointers : pointerChecks p ordinary.base=Bool.true)
    (profile : Profile) (_profileOK : profileCheck p ordinary.base profile=Bool.true)
    (args : Arguments p) (inputs : inputChecks p args=Bool.true) (before : State R) (alloc : Allocation p (admittedABI p args inputs) before)
    (μ : RegionName → Nat → ℝ) (h : Reads p before μ)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p (admittedABI p args inputs) before alloc work=some result ∧
      result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<96),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16 (some (expected p μ b h n d))) ∧
      (∀ r off,r≠p.Out → r≠alloc.unusedM.region → result.state.mem r off=before.mem r off) :=
  Real.correct p (admittedGuard p strides launch) storage (admittedABI p args inputs)
    before alloc μ h work legal

#axiomsClean checked_real
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
