import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Admission
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperProtocol

namespace VeriTile.Bench.Repairs.LayerNormFwd.Checked
open Wrapper Admission
noncomputable section
set_option maxHeartbeats 200000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

/-- Ordinary identity laws for the two removed tensor methods on exactly the
checked domain. These are local no-op laws, with no allocation, launch or output
claim. The actual checked wrapper below contains neither method call. -/
structure NoCopy (api : TensorAPI world) : Prop where
  resolve : ∀ role input s, input.negative = Bool.false →
    (api.resolve role input).run s = (s,.ok input)
  transfer : ∀ target role input s, input.device = target →
    (api.transfer target role input).run s = (s,.ok input)

/-- Trace-erased equality retains both failure state and result. It only erases
local no-op events from the retained historical host proof. -/
def Same {α : Type} (a b : Action world α) : Prop :=
  ∀ s, (a.run s).state = (b.run s).state ∧ (a.run s).result = (b.run s).result

 theorem same_refl (a : Action world α) : Same a a := fun _ => ⟨rfl,rfl⟩

 theorem same_bind_pure (a : Action world α) (value : α) (f : α → Action world β)
    (h : Same a (Action.pure value)) : Same (a.bind f) (f value) := by
  intro s
  obtain ⟨hs,hr⟩ := h s
  change (a.run s).state = s at hs
  change (a.run s).result = .ok value at hr
  simp only [Action.bind, hs, hr]
  constructor <;> trivial

 theorem same_bind (a b : Action world α) (f g : α → Action world β)
    (h : Same a b) (tail : ∀ value, Same (f value) (g value)) : Same (a.bind f) (b.bind g) := by
  intro s
  obtain ⟨hs,hr⟩ := h s
  cases hb : (b.run s).result with
  | error e => simp [Action.bind, hs, hr, hb]
  | ok value => simpa [Action.bind, hs, hr, hb] using tail value (b.run s).state

 theorem materialize_same (api : TensorAPI world) (hn : NoCopy api)
    (target : Nat) (role : Role) (input : Tensor)
    (negative : input.negative = Bool.false) (owner : input.device = target) :
    Same (materialize api target role input) (Action.pure input) := by
  intro s
  simp [materialize, bind, Action.bind, call, Action.pure,
    hn.resolve role input s negative, hn.transfer target role input s owner, Except.mapError]

 theorem optional_same (api : TensorAPI world) (hn : NoCopy api)
    (target : Nat) (role : Role) (input : Option Tensor)
    (h : ∀ t, input = some t → t.negative = Bool.false ∧ t.device = target) :
    Same (resolveOption api target role input) (Action.pure input) := by
  cases input with
  | none => exact same_refl _
  | some value =>
    have hstep := materialize_same api hn target role value (h value rfl).1 (h value rfl).2
    exact same_bind _ _ (fun t => Action.pure (some t)) (fun t => Action.pure (some t)) hstep
      (fun _ => same_refl _)

def inputs (q : Request) : Inputs :=
  ⟨q.x,q.x,q.weight,q.bias,q.residual,q.x1,q.weight1,q.bias1,q.rowscale⟩

 theorem optional_metadata (input : Option Input) (shape : List Nat) (strides : List Int) (device : Nat)
    (h : OptionalContract input shape strides device) :
    ∀ t, input.map Input.value = some t → t.negative = Bool.false ∧ t.device = device := by
  intro t ht
  cases hi : input with
  | none => simp [hi] at ht
  | some arg =>
    have he : arg.value = t := by simpa [hi] using ht
    rw [←he]
    exact ⟨(h arg hi).2.2.2.2.2.2.2.2.1,(h arg hi).2.2.2.2.1⟩

 theorem resolveInputs_same (api : TensorAPI world) (hn : NoCopy api)
    (c : Call) (h : InputDomain c) :
    Same (resolveInputs api (request c)) (Action.pure (inputs (request c))) := by
  obtain ⟨hx,hw,hb,hw1,hb1,hr,hx1,hrs,_⟩ := h
  have mx := materialize_same api hn c.x.value.device .x c.x.value hx.2.2.2.2.2.2.2.2.1 rfl
  have mw := materialize_same api hn c.x.value.device .weight c.weight.value
    hw.2.2.2.2.2.2.2.2.1 hw.2.2.2.2.1
  have mb := optional_same api hn c.x.value.device .bias _ (optional_metadata _ _ _ _ hb)
  have mr := optional_same api hn c.x.value.device .residual _ (optional_metadata _ _ _ _ hr)
  have mx1 := optional_same api hn c.x.value.device .x1 _ (optional_metadata _ _ _ _ hx1)
  have mw1 := optional_same api hn c.x.value.device .weight1 _ (optional_metadata _ _ _ _ hw1)
  have mb1 := optional_same api hn c.x.value.device .bias1 _ (optional_metadata _ _ _ _ hb1)
  have mrs := optional_same api hn c.x.value.device .rowscale _ (optional_metadata _ _ _ _ hrs)
  intro s
  have hmx := mx s; have hmw := mw s; have hmb := mb s; have hmr := mr s
  have hmx1 := mx1 s; have hmw1 := mw1 s; have hmb1 := mb1 s; have hmrs := mrs s
  simp only [Action.pure] at hmx hmw hmb hmr hmx1 hmw1 hmb1 hmrs
  simp only [resolveInputs, bind, Action.bind, note, request, hmx.1, hmx.2,
    hmw.1, hmw.2, hmb.1, hmb.2, hmr.1, hmr.2, hmx1.1, hmx1.2,
    hmw1.1, hmw1.2, hmb1.1, hmb1.2, hmrs.1, hmrs.2, pure, Action.pure, inputs]
  constructor <;> trivial

/-- Actual source prefix: validation, existing allocation calls, width check.
No resolve_neg, transfers or canonicalizing allocations are present. -/
def prepare (tensor : TensorAPI world) (q : Request) : Action world Prepared := do
  let validated ← validate q
  let input := inputs q
  let outputs ← allocateOutputs tensor q validated input
  let block ← width tensor validated input
  pure ⟨validated,input,outputs,block⟩

 theorem prepare_same (tensor : TensorAPI world) (hn : NoCopy tensor)
    (c : Call) (h : InputDomain c) : Same (Wrapper.prepare tensor (request c)) (prepare tensor (request c)) := by
  dsimp only [Wrapper.prepare, prepare, Bind.bind]
  apply same_bind _ _ _ _ (same_refl _)
  intro validated
  exact same_bind_pure (resolveInputs tensor (request c)) (inputs (request c))
    (fun input => do
      let output ← allocateOutputs tensor (request c) validated input
      let block ← width tensor validated input
      pure (⟨validated,input,output,block⟩ : Prepared))
    (resolveInputs_same tensor hn c h)

def surround (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) : Action world Returned := do
  let p ← prepare tensor q
  inDevice (device.enter p.inputs.x.device) (device.leave p.inputs.x.device) (do
    let args ← arguments device q p
    call .admission (device.admit args)
    note .execute
    kernel args)
  finish device p

 theorem surround_same (tensor : TensorAPI world) (hn : NoCopy tensor) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (c : Call) (h : InputDomain c) :
    Same (Wrapper.surround tensor device kernel (request c)) (surround tensor device kernel (request c)) := by
  unfold Wrapper.surround surround
  exact same_bind _ _ _ _ (prepare_same tensor hn c h) (fun _ => same_refl _)

def run (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (c : Call) (s : S) : Except Unit (Outcome S Returned) :=
  if checkInputs c then .ok ((surround tensor device kernel (request c)).run s) else .error ()

 theorem admitted (tensor : TensorAPI world) (hn : NoCopy tensor) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (c : Call) (s : S) (outcome : Outcome S Returned)
    (h : run tensor device kernel c s = .ok outcome) :
    InputDomain c ∧
    outcome.state = ((Wrapper.surround tensor device kernel (request c)).run s).state ∧
    outcome.result = ((Wrapper.surround tensor device kernel (request c)).run s).result := by
  unfold run at h
  split at h
  · rename_i hc
    have domain := (checkInputs_iff c).mp hc
    have he : ((surround tensor device kernel (request c)).run s) = outcome := Except.ok.inj h
    rw [←he]
    have hs := surround_same tensor hn device kernel c domain s
    exact ⟨domain,hs.1.symm,hs.2.symm⟩
  · cases h

#axiomsClean materialize_same
#axiomsClean resolveInputs_same
#axiomsClean prepare_same
#axiomsClean surround_same
#axiomsClean admitted
end
end VeriTile.Bench.Repairs.LayerNormFwd.Checked
