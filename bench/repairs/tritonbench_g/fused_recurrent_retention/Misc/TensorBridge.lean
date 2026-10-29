import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.fused_recurrent_retention.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.fused_recurrent_retention.Correct
import bench.repairs.tritonbench_g.fused_recurrent_retention.FPEquiv
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedCalls
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.CorrectPublic
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.FPEquivPublic

/-! Optional checked-call applications and actual Tensor byte/frame corollaries.
The complete quoted Real results remain in Correct; native observations are ordinary auxiliaries in NativeCorrect. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.TensorBridge
noncomputable section
open Numeric
local instance : Inhabited Word := ⟨VeriTile.Triton.FP.Value.ofBits .fp32 0⟩

theorem checked_quoted_returned (p : PlanBinding s) (x : Host.Forward ℝ s caller)
    (ok : x.check=Bool.true) :
    ∃ ret,Quoted.checkedForward realArithmetic p x=some ret ∧
      ForwardRealPost (x.call ok) ret.source ∧
      ret.output.identity=x.identities 2 ∧
      ret.output.view=ret.source.output.view ∧
      ret.final=ret.source.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1)) ∧
      ∀ (backCaller : Set Nat)
        (back : Host.Backward ℝ s ret.source.context x.before x.current backCaller)
        (backOK : back.check=Bool.true),
        ∃ grads,Quoted.checkedBackward realArithmetic p back=some grads ∧
          BackwardRealPost (back.call backOK) grads.source ∧
          grads.gradients=(Host.object (back.identities 3) (back.requiresGrad 3) grads.source.dq,
            Host.object (back.identities 4) (back.requiresGrad 4) grads.source.dk,
            Host.object (back.identities 5) (back.requiresGrad 5) grads.source.dv,none,none) := by
  let out := (x.call ok).sourceRun realArithmetic p
  refine ⟨⟨out,Host.object (x.identities 2) (x.requiresGrad 2) out.output,
    out.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1))⟩,?_,?_,rfl,rfl,rfl,?_⟩
  · rw [Quoted.checkedForward_eq]
    simp [Host.forward,ok,out]
  · dsimp [out]
    rw [ForwardCall.sourceRun_eq]
    exact forward_real p (x.call ok)
  · intro backCaller back backOK
    let grads := (back.call backOK).sourceRun realArithmetic p
    refine ⟨⟨grads,(Host.object (back.identities 3) (back.requiresGrad 3) grads.dq,
      Host.object (back.identities 4) (back.requiresGrad 4) grads.dk,
      Host.object (back.identities 5) (back.requiresGrad 5) grads.dv,none,none)⟩,?_,?_,rfl⟩
    · rw [Quoted.checkedBackward_eq realArithmetic p back (Quoted.savedDomain (x.call ok) _ _)]
      simp [Host.backward,backOK,grads]
    · dsimp [grads]
      rw [BackwardCall.sourceRun_eq]
      exact backward_real (x.call ok).scope.shape p (back.call backOK)

#axiomsClean checked_quoted_returned

theorem checked_quoted_fp_returned (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (x : Host.Forward Word s caller) (ok : x.check=Bool.true) :
    Quoted.checkedForward (nativeArithmetic ports) p x=Quoted.checkedForward (arithmetic u) p x ∧
    ∃ ret,Quoted.checkedForward (nativeArithmetic ports) p x=some ret ∧
      ForwardReferencePost (arithmetic u) p (x.call ok) ret.source ∧
      ret.output.identity=x.identities 2 ∧
      ret.output.view=ret.source.output.view ∧
      ∀ (backCaller : Set Nat)
        (back : Host.Backward Word s ret.source.context x.before x.current backCaller)
        (backOK : back.check=Bool.true),
        Quoted.checkedBackward (nativeArithmetic ports) p back=Quoted.checkedBackward (arithmetic u) p back ∧
        ∃ grads,Quoted.checkedBackward (nativeArithmetic ports) p back=some grads ∧
          BackwardReferencePost (arithmetic u) p (back.call backOK) grads.source ∧
          grads.gradients=(Host.object (back.identities 3) (back.requiresGrad 3) grads.source.dq,
            Host.object (back.identities 4) (back.requiresGrad 4) grads.source.dk,
            Host.object (back.identities 5) (back.requiresGrad 5) grads.source.dv,none,none) := by
  rw [nativeArithmetic_eq u ports laws]
  refine ⟨rfl,?_⟩
  let out := (x.call ok).sourceRun (arithmetic u) p
  refine ⟨⟨out,Host.object (x.identities 2) (x.requiresGrad 2) out.output,
    out.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1))⟩,?_,?_,rfl,rfl,?_⟩
  · rw [Quoted.checkedForward_eq]
    simp [Host.forward,ok,out]
  · dsimp [out]
    rw [ForwardCall.sourceRun_eq]
    exact forward_reference (arithmetic u) p (x.call ok)
  · intro backCaller back backOK
    refine ⟨rfl,?_⟩
    let grads := (back.call backOK).sourceRun (arithmetic u) p
    refine ⟨⟨grads,(Host.object (back.identities 3) (back.requiresGrad 3) grads.dq,
      Host.object (back.identities 4) (back.requiresGrad 4) grads.dk,
      Host.object (back.identities 5) (back.requiresGrad 5) grads.dv,none,none)⟩,?_,?_,rfl⟩
    · rw [Quoted.checkedBackward_eq (arithmetic u) p back (Quoted.savedDomain (x.call ok) _ _)]
      simp [Host.backward,backOK,grads]
    · dsimp [grads]
      rw [BackwardCall.sourceRun_eq]
      exact backward_reference (arithmetic u) p (back.call backOK)


#axiomsClean checked_quoted_fp_returned

/-- Guarded completed public call. Every structural tensor/index premise is
established by `check`; allocator/version observations remain runtime primitives.
The independent Real target includes the final-state cotangent and H0 detach. -/
theorem checked_returned (p : PlanBinding s) (x : Host.Forward ℝ s caller)
    (ok : x.check=Bool.true) :
    ∃ ret,Host.forward realArithmetic p x=some ret ∧
      ForwardRealPost (x.call ok) ret.source ∧
      ret.output.identity=x.identities 2 ∧
      ret.output.view=ret.source.output.view ∧
      ret.final=ret.source.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1)) ∧
      ∀ (backCaller : Set Nat)
        (back : Host.Backward ℝ s ret.source.context x.before x.current backCaller)
        (backOK : back.check=Bool.true),
        ∃ grads,Host.backward realArithmetic p back=some grads ∧
          BackwardRealPost (back.call backOK) grads.source ∧
          grads.gradients=(Host.object (back.identities 3) (back.requiresGrad 3) grads.source.dq,
            Host.object (back.identities 4) (back.requiresGrad 4) grads.source.dk,
            Host.object (back.identities 5) (back.requiresGrad 5) grads.source.dv,none,none) := by
  let out := (x.call ok).sourceRun realArithmetic p
  refine ⟨⟨out,Host.object (x.identities 2) (x.requiresGrad 2) out.output,
    out.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1))⟩,?_,?_,rfl,rfl,rfl,?_⟩
  · simp [Host.forward,ok,out]
  · dsimp [out]
    rw [ForwardCall.sourceRun_eq]
    exact forward_real p (x.call ok)
  · intro backCaller back backOK
    let grads := (back.call backOK).sourceRun realArithmetic p
    refine ⟨⟨grads,(Host.object (back.identities 3) (back.requiresGrad 3) grads.dq,
      Host.object (back.identities 4) (back.requiresGrad 4) grads.dk,
      Host.object (back.identities 5) (back.requiresGrad 5) grads.dv,none,none)⟩,?_,?_,rfl⟩
    · simp [Host.backward,backOK,grads]
    · dsimp [grads]
      rw [BackwardCall.sourceRun_eq]
      exact backward_real (x.call ok).scope.shape p (back.call backOK)

theorem checked_quoted_frame (p : PlanBinding s) (x : Host.Forward ℝ s caller)
    (address : Address) (outside : ¬CompleteFrame.forwardWrites s x.allocation x.args.outputFinal address) :
    Quoted.checkedForwardMemory realArithmetic p x address=x.before address := by
  rw [Quoted.checkedForwardMemory_eq]
  exact Host.forward_frame realArithmetic p x address outside

/-- Exact source-word result and actual returned objects. The seven scalar
laws are fixed-arity numerical leaves; no output, dot or Real equality is a premise. -/
theorem checked_fp_returned (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (x : Host.Forward Word s caller) (ok : x.check=Bool.true) :
    Host.forward (nativeArithmetic ports) p x=Host.forward (arithmetic u) p x ∧
    ∃ ret,Host.forward (nativeArithmetic ports) p x=some ret ∧
      ForwardReferencePost (arithmetic u) p (x.call ok) ret.source ∧
      ret.output.identity=x.identities 2 ∧
      ret.output.view=ret.source.output.view ∧
      ∀ (backCaller : Set Nat)
        (back : Host.Backward Word s ret.source.context x.before x.current backCaller)
        (backOK : back.check=Bool.true),
        Host.backward (nativeArithmetic ports) p back=Host.backward (arithmetic u) p back ∧
        ∃ grads,Host.backward (nativeArithmetic ports) p back=some grads ∧
          BackwardReferencePost (arithmetic u) p (back.call backOK) grads.source ∧
          grads.gradients=(Host.object (back.identities 3) (back.requiresGrad 3) grads.source.dq,
            Host.object (back.identities 4) (back.requiresGrad 4) grads.source.dk,
            Host.object (back.identities 5) (back.requiresGrad 5) grads.source.dv,none,none) := by
  rw [nativeArithmetic_eq u ports laws]
  refine ⟨rfl,?_⟩
  let out := (x.call ok).sourceRun (arithmetic u) p
  refine ⟨⟨out,Host.object (x.identities 2) (x.requiresGrad 2) out.output,
    out.finalState.map (Host.object (x.identities 1) (x.requiresGrad 1))⟩,?_,?_,rfl,rfl,?_⟩
  · simp [Host.forward,ok,out]
  · dsimp [out]
    rw [ForwardCall.sourceRun_eq]
    exact forward_reference (arithmetic u) p (x.call ok)
  · intro backCaller back backOK
    refine ⟨rfl,?_⟩
    let grads := (back.call backOK).sourceRun (arithmetic u) p
    refine ⟨⟨grads,(Host.object (back.identities 3) (back.requiresGrad 3) grads.dq,
      Host.object (back.identities 4) (back.requiresGrad 4) grads.dk,
      Host.object (back.identities 5) (back.requiresGrad 5) grads.dv,none,none)⟩,?_,?_,rfl⟩
    · simp [Host.backward,backOK,grads]
    · dsimp [grads]
      rw [BackwardCall.sourceRun_eq]
      exact backward_reference (arithmetic u) p (back.call backOK)

/-- Whole native memory equality also fixes every observed byte, including
source scratch. Rejection and the global complement frame are unconditional. -/
theorem checked_fp_bytes (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (x : Host.Forward Word s caller) :
    CompleteFrame.byteView (Host.forwardMemory (nativeArithmetic ports) p x)=
      CompleteFrame.byteView (Host.forwardMemory (arithmetic u) p x) := by
  rw [nativeArithmetic_eq u ports laws]

theorem checked_quoted_fp_bytes (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (x : Host.Forward Word s caller) :
    CompleteFrame.byteView (Quoted.checkedForwardMemory (nativeArithmetic ports) p x)=
      CompleteFrame.byteView (Host.forwardMemory (arithmetic u) p x) := by
  rw [Quoted.checkedForwardMemory_eq,nativeArithmetic_eq u ports laws]

theorem checked_quoted_fp_frame (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (x : Host.Forward Word s caller) (address : Address)
    (outside : ¬CompleteFrame.forwardWrites s x.allocation x.args.outputFinal address) :
    Quoted.checkedForwardMemory (nativeArithmetic ports) p x address=x.before address := by
  rw [Quoted.checkedForwardMemory_eq]
  exact Host.forward_frame (nativeArithmetic ports) p x address outside

theorem checked_quoted_backward_bytes (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (p : PlanBinding s) (origin : ForwardCall Word s caller)
    (back : Host.Backward Word s origin.args.saved snapshot current backCaller) :
    CompleteFrame.byteView (Quoted.checkedBackwardMemory (nativeArithmetic ports) p back)=
      CompleteFrame.byteView (Host.backwardMemory (arithmetic u) p back) := by
  rw [Quoted.checkedBackwardMemory_eq _ p back (Quoted.savedDomain origin _ _),nativeArithmetic_eq u ports laws]

theorem checked_quoted_backward_frame (ports : ScalarPorts) (p : PlanBinding s)
    (origin : ForwardCall Word s caller)
    (back : Host.Backward Word s origin.args.saved snapshot current backCaller) (address : Address)
    (outside : ¬CompleteFrame.backwardWrites s back.allocation address) :
    Quoted.checkedBackwardMemory (nativeArithmetic ports) p back address=back.before address := by
  rw [Quoted.checkedBackwardMemory_eq _ p back (Quoted.savedDomain origin _ _)]
  exact Host.backward_frame (nativeArithmetic ports) p back address outside

#axiomsClean checked_returned
#axiomsClean checked_quoted_frame
#axiomsClean checked_fp_returned
#axiomsClean checked_fp_bytes
#axiomsClean checked_quoted_fp_bytes
#axiomsClean checked_quoted_fp_frame
#axiomsClean checked_quoted_backward_bytes
#axiomsClean checked_quoted_backward_frame
#axiomsClean Host.forward_rejected
#axiomsClean Host.backward_rejected
#axiomsClean Host.forward_frame
#axiomsClean Host.backward_frame
#axiomsClean CompleteFrame.forward_bytes
#axiomsClean CompleteFrame.backward_bytes
#axiomsClean Host.forward_bytes
#axiomsClean Host.backward_bytes
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.TensorBridge
