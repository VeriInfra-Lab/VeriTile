import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Scope
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.RawFrame
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.FinalMath
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

def ForwardCall.run (call : ForwardCall α s caller) (a : Arithmetic α) (p : PlanBinding s) : ForwardReturn α :=
  executeForward a s p.plans call.args call.allocation call.before

def BackwardCall.buffers (call : BackwardCall α s ctx snapshot current caller) : InputBuffers :=
  ctx.buffers call.do_.view (call.final.map TensorObject.view)

def BackwardCall.run (call : BackwardCall α s ctx snapshot current caller) (a : Arithmetic α)
    (p : PlanBinding s) : BackwardReturn α :=
  executeBackward a s p.plans ctx call.do_.view (call.final.map TensorObject.view) call.allocation call.before

def ForwardMetadata (call : ForwardCall α s caller) (ret : ForwardReturn α) : Prop :=
  ret.output=resultTensor (call.allocation.storage 2) call.args.q.view.device s.B s.H s.T s.DV ∧
  ret.finalState=(if call.args.outputFinal then some (resultTensor (call.allocation.storage 1)
    call.args.q.view.device s.B s.H s.DK s.DV) else none) ∧
  ret.context=call.args.saved ∧ ret.output.view.storage∉caller

def BackwardMetadata (call : BackwardCall α s ctx snapshot current caller) (ret : BackwardReturn α) : Prop :=
  ret.dq=resultTensor (call.allocation.storage 3) ctx.q.view.device s.B s.H s.T s.DK ∧
  ret.dk=resultTensor (call.allocation.storage 4) ctx.q.view.device s.B s.H s.T s.DK ∧
  ret.dv=resultTensor (call.allocation.storage 5) ctx.q.view.device s.B s.H s.T s.DV ∧
  ret.initialGradient=none ∧ ret.outputFlagGradient=none ∧
  ret.dq.view.storage∉caller ∧ ret.dk.view.storage∉caller ∧ ret.dv.view.storage∉caller

def ForwardReferencePost (a : Arithmetic α) (p : PlanBinding s) (call : ForwardCall α s caller)
    (ret : ForwardReturn α) : Prop :=
  ForwardMetadata call ret ∧
  (∀ h<s.BH,∀ t<s.T,∀ v<s.DV,ret.memory (call.allocation.storage 2,(rowAddress s s.DV h t v : Int))=
    resultO a s p.plans (call.args.buffers.observe s call.before) call.args.seed.isSome h t v) ∧
  (call.args.outputFinal=Bool.true → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    ret.memory (call.allocation.storage 1,(stateOffset s h k v : Int))=
      state a s (call.args.buffers.observe s call.before) call.args.seed.isSome h k v s.T) ∧
  (∀ address,address.1∈caller → ret.memory address=call.before address)

def BackwardReferencePost (a : Arithmetic α) (p : PlanBinding s)
    (call : BackwardCall α s ctx snapshot current caller) (ret : BackwardReturn α) : Prop :=
  BackwardMetadata call ret ∧
  (∀ h<s.BH,∀ t<s.T,∀ k<s.DK,ret.memory (call.allocation.storage 3,(rowAddress s s.DK h t k : Int))=
    resultQ a s p.plans (call.buffers.observe s call.before) ctx.initial.isSome h t k) ∧
  (∀ h<s.BH,∀ t<s.T,∀ k<s.DK,ret.memory (call.allocation.storage 4,(rowAddress s s.DK h t k : Int))=
    resultK a s p.plans (call.buffers.observe s call.before) call.final.isSome h t k) ∧
  (∀ h<s.BH,∀ t<s.T,∀ v<s.DV,ret.memory (call.allocation.storage 5,(rowAddress s s.DV h t v : Int))=
    resultV a s p.plans (call.buffers.observe s call.before) call.final.isSome h t v) ∧
  (∀ address,address.1∈caller → ret.memory address=call.before address)

theorem forward_reference (a : Arithmetic α) (p : PlanBinding s) (call : ForwardCall α s caller) :
    ForwardReferencePost a p call (call.run a p) := by
  refine ⟨⟨rfl,rfl,rfl,call.allocation.fresh 2⟩,?_,?_,?_⟩
  · intro h hh t ht v hv
    exact forward_output a s p.plans p.valid _ _ _ _ _ h t v hh ht hv
  · intro hf h hh k hk v hv
    change forwardRun a s p.plans call.args.buffers call.allocation call.args.seed.isSome call.args.outputFinal _ _=_
    rw [hf]
    exact forward_state a s p.plans _ _ call.inputs _ _ h k v hh hk hv
  · intro address ha
    exact forward_frame a s p.plans _ _ _ _ _ address ha

theorem backward_reference (a : Arithmetic α) (p : PlanBinding s)
    (call : BackwardCall α s ctx snapshot current caller) :
    BackwardReferencePost a p call (call.run a p) := by
  have hm : (call.final.map TensorObject.view).isSome=call.final.isSome := by simp
  refine ⟨⟨rfl,rfl,rfl,rfl,rfl,call.allocation.fresh 3,call.allocation.fresh 4,call.allocation.fresh 5⟩,?_,?_,?_,?_⟩
  · intro h hh t ht k hk
    exact backward_dq a s p.plans p.valid _ _ _ _ _ h t k hh ht hk
  · intro h hh t ht k hk
    simpa [BackwardCall.run,executeBackward,hm] using
      backward_dk a s p.plans p.valid call.buffers call.allocation call.inputs ctx.initial.isSome
        (call.final.map TensorObject.view).isSome call.before h t k hh ht hk
  · intro h hh t ht v hv
    simpa [BackwardCall.run,executeBackward,hm] using
      backward_dv a s p.plans p.valid call.buffers call.allocation call.inputs ctx.initial.isSome
        (call.final.map TensorObject.view).isSome call.before h t v hh ht hv
  · intro address ha
    exact backward_frame a s p.plans _ _ _ _ _ address ha

def ForwardRealPost (call : ForwardCall ℝ s caller) (ret : ForwardReturn ℝ) : Prop :=
  ForwardMetadata call ret ∧
  (∀ h<s.BH,∀ t<s.T,∀ v<s.DV,ret.memory (call.allocation.storage 2,(rowAddress s s.DV h t v : Int))=
    mathOutput s (call.args.buffers.observe s call.before) call.args.seed.isSome h t v) ∧
  (call.args.outputFinal=Bool.true → ∀ h<s.BH,∀ k<s.DK,∀ v<s.DV,
    ret.memory (call.allocation.storage 1,(stateOffset s h k v : Int))=
      mathF s (call.args.buffers.observe s call.before) call.args.seed.isSome h k v) ∧
  (∀ address,address.1∈caller → ret.memory address=call.before address)

def BackwardRealPost (call : BackwardCall ℝ s ctx snapshot current caller) (ret : BackwardReturn ℝ) : Prop :=
  BackwardMetadata call ret ∧
  (∀ h<s.BH,∀ t<s.T,∀ k<s.DK,ret.memory (call.allocation.storage 3,(rowAddress s s.DK h t k : Int))=
    mathDQ s (call.buffers.observe s call.before) ctx.initial.isSome h t k) ∧
  (∀ h<s.BH,∀ t<s.T,∀ k<s.DK,ret.memory (call.allocation.storage 4,(rowAddress s s.DK h t k : Int))=
    mathDK s (call.buffers.observe s call.before) call.final.isSome h t k) ∧
  (∀ h<s.BH,∀ t<s.T,∀ v<s.DV,ret.memory (call.allocation.storage 5,(rowAddress s s.DV h t v : Int))=
    mathDV s (call.buffers.observe s call.before) call.final.isSome h t v) ∧
  (∀ address,address.1∈caller → ret.memory address=call.before address)

theorem forward_real (p : PlanBinding s) (call : ForwardCall ℝ s caller) :
    ForwardRealPost call (call.run realArithmetic p) := by
  obtain ⟨hm,ho,hf,frame⟩ := forward_reference realArithmetic p call
  refine ⟨hm,?_,?_,frame⟩
  · intro h hh t ht v hv
    rw [ho h hh t ht v hv]
    exact resultO_real s call.scope.shape p.plans p.valid _ _ h t v hv
  · intro present h hh k hk v hv
    rw [hf present h hh k hk v hv]
    exact state_real s _ _ h k v s.T hk hv

theorem backward_real (shape : s.Admitted) (p : PlanBinding s)
    (call : BackwardCall ℝ s ctx snapshot current caller) :
    BackwardRealPost call (call.run realArithmetic p) := by
  obtain ⟨hm,hq,hk,hv,frame⟩ := backward_reference realArithmetic p call
  refine ⟨hm,?_,?_,?_,frame⟩
  · intro h hh t ht k hkey
    rw [hq h hh t ht k hkey]
    exact resultQ_real s shape p.plans p.valid _ _ h t k hkey
  · intro h hh t ht k hkey
    rw [hk h hh t ht k hkey]
    exact resultK_real s shape p.plans p.valid _ _ h t k hkey ht
  · intro h hh t ht v hval
    rw [hv h hh t ht v hval]
    exact resultV_real s shape p.plans p.valid _ _ h t v hval ht

#axiomsClean forward_reference
#axiomsClean backward_reference
#axiomsClean forward_real
#axiomsClean backward_real
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
