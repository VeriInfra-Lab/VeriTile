import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStreams

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStream
open Wrapper WrapperNative
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}
variable {tensor : TensorAPI world} {device : DispatchAPI world}

def KeepsOwner (api : API tensor device) (action : Action world α) : Prop :=
  ∀ s, api.activeOwner (action.run s).state=api.activeOwner s

theorem keeps_bind (api : API tensor device) (a : Action world α) (f : α → Action world β)
    (ha : KeepsOwner api a) (hf : ∀ value, KeepsOwner api (f value)) : KeepsOwner api (a.bind f) := by
  intro s
  cases h : (a.run s).result with
  | error e => simpa [Action.bind,h] using ha s
  | ok value => simpa [Action.bind,h] using (hf value (a.run s).state).trans (ha s)

theorem keeps_optionalStride (api : API tensor device) (role : Role) (value : Option Tensor) :
    KeepsOwner api (optionalStride role value) := by
  cases value <;> intro s <;> rfl

theorem arguments_owner (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (q : Request) (p : Prepared) (s : S) :
    api.activeOwner (argumentsState device q p s)=api.activeOwner s := by
  suffices h : KeepsOwner api (arguments device q p) from h s
  unfold arguments
  apply keeps_bind api
  · intro state
    exact api.select_owner [p.validated.M] state
  intro _
  apply keeps_bind api
  · exact fun _ => rfl
  intro _
  apply keeps_bind api
  · exact fun _ => rfl
  intro _
  apply keeps_bind api
  · exact keeps_optionalStride api .residual p.inputs.residual
  intro _
  apply keeps_bind api
  · exact keeps_optionalStride api .residualOut p.outputs.residualOut
  intro _
  apply keeps_bind api
  · exact keeps_optionalStride api .x1 p.inputs.x1
  intro _
  apply keeps_bind api
  · exact keeps_optionalStride api .y1 p.outputs.y1
  intro _
  exact fun _ => rfl

/-- Every active read tensor is on the selected device, with its producer
ordered before this invocation's actual Triton launch stream. RNG seeds are
covered by the independent ordinary randint producer contract. -/
def LaunchReady (api : API tensor device) (q : Request) (p : Prepared) (s : S) : Prop :=
  api.activeOwner s=q.x.device ∧ api.activeStream s=api.current q.x.device ∧
  InputsReady api q.x.device p.inputs ∧ Optional api q.x.device p.outputs.seeds

theorem prefix_ready (tensor : TensorAPI world) (device : DispatchAPI world)
    (api : API tensor device) (q : Request) (initial : S) (p : Prepared) (args : LaunchArguments)
    (route : LaunchPrefix tensor device q initial p args) :
    LaunchReady api q p (launchState tensor device q initial p args) := by
  obtain ⟨hinputs,hseeds⟩ := prepare_ready tensor device api q initial p route.prepared
  have howner : api.activeOwner (launchState tensor device q initial p args)=q.x.device := by
    unfold launchState admittedState
    rw [api.admit_owner,arguments_owner tensor device api]
    exact (api.enter_owner p.inputs.x.device _ route.entered).trans hinputs.x.1
  exact ⟨howner,(api.active_stream _).trans (congrArg api.current howner),hinputs,hseeds⟩

theorem optional_of_some (api : API tensor device) (target : Nat) (input : Option Tensor)
    (h : Optional api target input) (value : Tensor) (he : input=some value) :
    Local api target value := by
  simpa only [he,Optional] using h

/-- These are precisely the source load roles, including both seed offsets
and both rowscale offsets. The tensor is the actual launch role tensor. -/
theorem read_role_ready (api : API tensor device) (q : Request) (p : Prepared) (s : S)
    (h : LaunchReady api q p s) (role : LayerNormFwd.Address.Role)
    (hread : ¬NativeMachineMemory.writeRole role) (value : Tensor)
    (he : roleTensor p role=some value) :
    value.device=api.activeOwner s ∧ api.ready value (api.activeStream s) := by
  have hlocal : Local api q.x.device value := by
    cases role <;> simp only [NativeMachineMemory.writeRole,not_true_eq_false] at hread
    case x =>
      have he' : p.inputs.x=value := Option.some.inj he
      exact he' ▸ h.2.2.1.x
    case weight =>
      have he' : p.inputs.weight=value := Option.some.inj he
      exact he' ▸ h.2.2.1.weight
    case residual => exact optional_of_some api _ _ h.2.2.1.residual value he
    case x1 => exact optional_of_some api _ _ h.2.2.1.x1 value he
    case bias => exact optional_of_some api _ _ h.2.2.1.bias value he
    case weight1 => exact optional_of_some api _ _ h.2.2.1.weight1 value he
    case bias1 => exact optional_of_some api _ _ h.2.2.1.bias1 value he
    case rowscale | rowscale1 => exact optional_of_some api _ _ h.2.2.1.rowscale value he
    case seeds | seeds1 => exact optional_of_some api _ _ h.2.2.2 value he
  exact ⟨hlocal.1.trans h.1.symm,by rw [h.2.1]; exact hlocal.2⟩

/-- The real transfer failure is retained, after a successful resolve, with
its state and exact two-step trace. No allocation or launch follows it. -/
theorem materialize_transfer_failure (tensor : TensorAPI world) (target : Nat) (role : Role)
    (input : Tensor) (initial : S) (resolved : Tensor) (code : Nat)
    (hr : ((tensor.resolve role input).run initial).2=.ok resolved)
    (ht : ((tensor.transfer target role resolved).run ((tensor.resolve role input).run initial).1).2=.error code) :
    ((materialize tensor target role input).run initial)=
      ⟨((tensor.transfer target role resolved).run ((tensor.resolve role input).run initial).1).1,
        .error (.api (.transfer role) code),[.resolve role,.transfer role]⟩ := by
  simp [materialize,Bind.bind,Action.bind,call,hr,ht,Except.mapError]

#axiomsClean keeps_bind
#axiomsClean keeps_optionalStride
#axiomsClean arguments_owner
#axiomsClean prefix_ready
#axiomsClean optional_of_some
#axiomsClean read_role_ready
#axiomsClean materialize_transfer_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStream
