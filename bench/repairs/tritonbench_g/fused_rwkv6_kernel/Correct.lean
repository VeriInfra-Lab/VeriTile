import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Checked
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.HistoricalCorrect
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Correct
noncomputable section
open Checked

def invoke (c : Shape) (x : Inputs) (plan : Plans c) (scale : ℝ)
    (pythonScale : PythonNumber) (caller : VeriTile.Torch.Device) (base : Region → Nat)
    (alloc : Allocations c x base) (memory : Memory ℝ) : Option (Returned (Memory ℝ)) :=
  if admit c x then some (returned c x base alloc pythonScale caller
    (forward realOps c plan scale memory)) else none

/-- Complete exact-real recurrence, actual returned Tensor metadata, optional
final state, original saved Tensor handles/scale/reverse and the whole frame.
Native FP32 conformance is an independent endpoint. -/
theorem rwkv_real (c : Shape) (x : Inputs) (plan : Plans c) (scale : ℝ)
    (pythonScale : PythonNumber) (caller : VeriTile.Torch.Device) (base : Region → Nat)
    (alloc : Allocations c x base) (memory : Memory ℝ) (admitted : admit c x=true) :
    ∃ result,invoke c x plan scale pythonScale caller base alloc memory=some result ∧
      result.output=alloc.output ∧ result.output.Valid ∧
      result.output.metadata.shape=tensorShape c .result ∧
      result.output.metadata.device=(x.tensor .q).metadata.device ∧
      result.output.metadata.dataPtr.toNat=base .result ∧
      result.finalState=(if c.final then some alloc.terminal else none) ∧
      (c.final=true → alloc.terminal.Valid ∧ alloc.terminal.metadata.shape=tensorShape c .ht ∧
        alloc.terminal.metadata.device=(x.tensor .q).metadata.device ∧ alloc.terminal.metadata.dataPtr.toNat=base .ht) ∧
      result.saved=saved x ∧ result.savedScale=pythonScale ∧
      result.savedReverse=c.reverse ∧ result.callerDevice=caller ∧
      (∀ z<c.Z,∀ t<c.T,∀ f<c.V,
        result.memory .result (resultAddress c z t f)=realOutput c (inputs c memory) scale z (visit c t) f) ∧
      (c.final=true → ∀ z<c.Z,∀ e<c.K,∀ f<c.V,
        result.memory .ht (stateAddress c z e f)=realState c (inputs c memory) z e f c.T) ∧
      (∀ r b,¬WholeOutput c r b → result.memory r b=memory r b) := by
  have hc := domain c x admitted
  refine ⟨returned c x base alloc pythonScale caller (forward realOps c plan scale memory),
    by simp only [invoke,admitted,if_true],rfl,alloc.outputMetadata.1,
    alloc.outputMetadata.2.1,alloc.outputMetadata.2.2.1,alloc.outputMetadata.2.2.2,
    rfl,alloc.terminalMetadata,rfl,rfl,rfl,rfl,?_,?_,?_⟩
  · intro z hz t ht f hf
    exact real_returned c hc.sizes plan scale memory z t f hz ht hf
  · intro hs z hz e he f hf
    exact real_terminal c hc.sizes plan scale memory hs z e f hz he hf
  · exact complete_frame realOps c hc.sizes plan scale memory

#axiomsClean rwkv_real
#specNonCircular realOutput avoiding [invoke,runLoop,runProgram,runGrid,forward,wrapper]
#specNonCircular realState avoiding [invoke,runLoop,runProgram,runGrid,forward,wrapper]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Correct
