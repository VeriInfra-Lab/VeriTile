import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelForward
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelNativeMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNative
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelNativeCarrier
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

inductive ForwardExpected (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) : RegionName → Nat → Word → Prop where
  | output (t : Fin s.T) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
      ForwardExpected isa routing scale s f bh iv u "o" (valAddress s bh iv s.FBV t.val j.val)
        ((KernelNativeSpec.output routing s (KernelNativeSpec.forwardInput isa routing s f bh iv u) scale t.val j).eval isa routing)
  | residual (t : Fin s.T) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
      ForwardExpected isa routing scale s f bh iv u "v" (valAddress s bh iv s.FBV t.val j.val)
        ((KernelNativeSpec.residual routing s (KernelNativeSpec.forwardInput isa routing s f bh iv u) t.val j).eval isa routing)
  | terminal (enabled : f.final=Bool.true) (i : Fin s.BK) (j : Fin s.FBV)
      (key : i.val<s.K) (value : iv*s.FBV+j.val<s.V) :
      ForwardExpected isa routing scale s f bh iv u "ht" (stateAddress s bh iv s.FBV i.val j.val)
        ((KernelNativeSpec.forward routing s (KernelNativeSpec.forwardInput isa routing s f bh iv u) s.T (j,i,PUnit.unit)).eval isa routing)

inductive BackwardExpected (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) : RegionName → Nat → Word → Prop where
  | query (t : Fin s.T) (i : Fin s.BK) (active : i.val<s.K) :
      BackwardExpected isa routing scale s f bh iv u "dq" (keyAddress s (plane s bh iv) t.val i.val)
        ((KernelNativeMemory.query isa routing scale s f bh iv u t.val i).eval isa routing)
  | key (t : Fin s.T) (i : Fin s.BK) (active : i.val<s.K) :
      BackwardExpected isa routing scale s f bh iv u "dk" (keyAddress s (plane s bh iv) t.val i.val)
        ((KernelNativeMemory.corrected isa routing scale s f bh iv u t.val i).eval isa routing)
  | value (t : Fin s.T) (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
      BackwardExpected isa routing scale s f bh iv u "dv" (valAddress s bh iv s.BBV t.val j.val)
        ((KernelNativeBackwardSpec.dv routing s (KernelNativeBackwardSpec.input isa routing s f bh iv u) t.val
          (KernelNativeMemory.enhanced isa routing scale s f bh iv u (s.T-1-t.val)) j).eval isa routing)
  | vectorBeta (enabled : f.vectorBeta=Bool.true) (t : Fin s.T) (j : Fin s.BBV) (active : iv*s.BBV+j.val<s.V) :
      BackwardExpected isa routing scale s f bh iv u "dbeta" (valAddress s (plane s bh iv) iv s.BBV t.val j.val)
        ((KernelNativeBackwardSpec.dbeta routing s (KernelNativeBackwardSpec.input isa routing s f bh iv u) t.val
          (KernelNativeMemory.enhanced isa routing scale s f bh iv u (s.T-1-t.val)) j).eval isa routing)
  | scalarBeta (enabled : f.vectorBeta=Bool.false) (t : Fin s.T) :
      BackwardExpected isa routing scale s f bh iv u "dbeta" (plane s bh iv*s.T+t.val)
        ((KernelNativeBackwardSpec.betaSum routing s
          (KernelNativeBackwardSpec.dbeta routing s (KernelNativeBackwardSpec.input isa routing s f bh iv u) t.val
            (KernelNativeMemory.enhanced isa routing scale s f bh iv u (s.T-1-t.val)))).eval isa routing)
  | seed (enabled : f.initialGrad=Bool.true) (i : Fin s.BK) (j : Fin s.BBV)
      (key : i.val<s.K) (value : iv*s.BBV+j.val<s.V) :
      BackwardExpected isa routing scale s f bh iv u "dh0" (stateAddress s bh iv s.BBV i.val j.val)
        ((KernelNativeBackwardSpec.reverse routing s scale (KernelNativeBackwardSpec.input isa routing s f bh iv u)
          s.T (i,j,PUnit.unit)).eval isa routing)

theorem forward_complete (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,KernelForward.Facts (algebra isa routing scale) s f scaleSlot bh iv u v ∧
      ∀ r off word,ForwardExpected isa routing scale s f bh iv u r off word → observe isa routing (v.mem r off)=word := by
  obtain ⟨v,facts⟩ := KernelForward.complete (algebra isa routing scale) s f scaleSlot bh iv u hv hk hh
  refine ⟨v,facts,?_⟩
  intro r off word expected
  cases expected with
  | output t j active =>
    rw [facts.outputs t j active,stored_observation,KernelNativeSpec.input_lowering,KernelNativeSpec.output_lowering]
  | residual t j active =>
    rw [facts.residuals t j active,stored_observation,KernelNativeSpec.input_lowering,KernelNativeSpec.residual_lowering]
  | terminal enabled i j key value =>
    rw [facts.terminal enabled i j key value,stored_observation]
    unfold KernelForwardLoop.carry
    rw [KernelNativeSpec.input_lowering,KernelNativeSpec.forward_lowering]

theorem backward_complete (isa : Instructions) (routing : Routing) (scale : Word) (s : Shape) (f : Flags)
    (bh iv : Nat) (u : State Node) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh) :
    ∃ v,KernelBackwardFacts.Facts (algebra isa routing scale) s f scaleSlot bh iv u v ∧
      ∀ r off word,BackwardExpected isa routing scale s f bh iv u r off word → observe isa routing (v.mem r off)=word := by
  obtain ⟨v,facts⟩ := KernelBackwardFacts.complete (algebra isa routing scale) s f scaleSlot bh iv u hv hk hh
  refine ⟨v,facts,?_⟩
  intro r off word expected
  cases expected with
  | query t i active => rw [facts.query t i active,stored_observation,KernelNativeMemory.query_lowering]
  | key t i active => rw [facts.key t i active,stored_observation,KernelNativeMemory.corrected_lowering]
  | value t j active =>
    rw [facts.value t j active,stored_observation]
    unfold KernelBackwardFacts.value KernelBackwardFacts.enhancedAt
    rw [KernelNativeMemory.enhanced_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.dv_lowering]
  | vectorBeta enabled t j active =>
    rw [facts.vectorBeta enabled t j active,stored_observation]
    unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
    rw [KernelNativeMemory.enhanced_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.dbeta_lowering]
  | scalarBeta enabled t =>
    rw [facts.scalarBeta enabled t,stored_observation]
    unfold KernelBackwardFacts.beta KernelBackwardFacts.enhancedAt
    rw [KernelNativeMemory.enhanced_lowering,KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.dbeta_lowering]
    rfl
  | seed enabled i j key value =>
    rw [facts.seed enabled i j key value,stored_observation]
    unfold KernelReverseMemory.carry
    rw [KernelNativeBackwardSpec.input_lowering,KernelNativeBackwardSpec.reverse_lowering]

#axiomsClean forward_complete
#axiomsClean backward_complete
#specNonCircular ForwardExpected avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelForward.result]
#specNonCircular BackwardExpected avoiding [KernelProgram.source,FP.Structural.exec,FP.Structural.run,KernelBackward.result]
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelNative
