import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPreparation

/-! The actual four gradient slots and ordinary view/device-return operations.
The epsilon slot is None. Transfers may alias their source when the owner is
unchanged; only logical values and the actual result identity are specified. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type}

abbrev Gradients := Storage.Tensor × Storage.Tensor × Option Storage.Tensor × Option Storage.Tensor

/-- Retained helper name; the source now returns only a view, with no transfer. -/
def transferView (lib : APIHost api neg inputs Metadata) (t : Dense) (shape : List Nat) (_device : Nat) :
    Action (Host.world (Metadata:=Metadata) api inputs) Storage.Tensor :=
  (call .returnTuple (lib.view t shape).call).bind fun viewed => Action.pure viewed.tensor

structure TransferRoute (lib : APIHost api neg inputs Metadata) (t : Dense) (shape : List Nat) (_device : Nat)
    (initial : State Store Metadata) where
  viewed : Dense
  viewOK : ((lib.view t shape).call.run initial).2=.ok viewed

variable {lib : APIHost api neg inputs Metadata} {t : Dense} {shape : List Nat} {device : Nat}
  {initial : State Store Metadata}

def TransferRoute.output (r : TransferRoute lib t shape device initial) := r.viewed.tensor
def TransferRoute.sView (_ : TransferRoute lib t shape device initial) := ((lib.view t shape).call.run initial).1
def TransferRoute.final (r : TransferRoute lib t shape device initial) := r.sView
def TransferRoute.transition (r : TransferRoute lib t shape device initial) : Transition api neg inputs initial r.final :=
  (lib.view t shape).transition initial

theorem transfer_route (out : Storage.Tensor)
    (ok : ((transferView lib t shape device).run initial).result=.ok out) :
    ∃ r : TransferRoute lib t shape device initial,r.output=out ∧
      ((transferView lib t shape device).run initial).state=r.final := by
  obtain ⟨viewed,hv,ht,hs,_⟩ := bind_success _ _ _ _ ok
  have eq : viewed.tensor=out := by simpa [Action.pure] using ht
  exact ⟨⟨viewed,(call_ok _ _ _ _).mp hv⟩,eq,hs⟩

theorem transfer_metadata (r : TransferRoute lib t shape device initial) :
    r.output.view.shape=shape ∧ r.output.view.device=t.tensor.view.device ∧ r.final.live r.output := by
  have hv := lib.viewed t shape initial r.viewed r.viewOK
  refine ⟨hv.shape,?_,hv.present⟩
  change r.viewed.tensor.view.device=t.tensor.view.device
  rw [r.viewed.tensor.owner,t.tensor.owner,hv.backing]

theorem transfer_values (r : TransferRoute lib t shape device initial) (i : Nat) (hi : i<t.count) :
    Storage.logical api neg r.final.memory r.output i=Storage.logical api neg initial.memory t.tensor i :=
  (lib.viewed t shape initial r.viewed r.viewOK).values i hi

theorem transfer_real (r : TransferRoute lib t shape device initial)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<t.count) :
    Storage.realLogical (r.transition.shadow m) r.output i=Storage.realLogical m t.tensor i :=
  (lib.viewed t shape initial r.viewed r.viewOK).realValues m i hi

def optionalReturn (lib : APIHost api neg inputs Metadata) (b : Option Dense)
    (shape : Option (List Nat)) (device : Option Nat) :
    Action (Host.world (Metadata:=Metadata) api inputs) (Option Storage.Tensor) :=
  match b with
  | none => Action.pure none
  | some b => (transferView lib b (shape.getD []) (device.getD 0)).bind fun result => Action.pure (some result)

inductive OptionalReturn (lib : APIHost api neg inputs Metadata)
    (shape : Option (List Nat)) (device : Option Nat) (initial : State Store Metadata) : Option Dense → Type
  | none : OptionalReturn lib shape device initial none
  | some {b : Dense} (route : TransferRoute lib b (shape.getD []) (device.getD 0) initial) :
      OptionalReturn lib shape device initial (some b)

def OptionalReturn.output {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat} :
    OptionalReturn lib shape device initial b → Option Storage.Tensor
  | .none => Option.none | .some r => Option.some r.output

def OptionalReturn.final {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat} :
    OptionalReturn lib shape device initial b → State Store Metadata
  | .none => initial | .some r => r.final

def OptionalReturn.transition {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat}
    (r : OptionalReturn lib shape device initial b) : Transition api neg inputs initial r.final :=
  match r with | .none => .refl initial | .some r => r.transition

theorem optional_route (b : Option Dense) (shape : Option (List Nat)) (device : Option Nat) (out : Option Storage.Tensor)
    (ok : ((optionalReturn lib b shape device).run initial).result=.ok out) :
    ∃ r : OptionalReturn lib shape device initial b,r.output=out ∧
      ((optionalReturn lib b shape device).run initial).state=r.final := by
  cases b with
  | none =>
    have h : out=none := by simpa [optionalReturn,Action.pure] using ok.symm
    subst out
    exact ⟨.none,rfl,rfl⟩
  | some b =>
    obtain ⟨d,hd,ho,hs,_⟩ := bind_success _ _ _ _ ok
    obtain ⟨r,he,hstate⟩ := transfer_route d hd
    subst d
    have h : out=some r.output := by simpa [Action.pure] using ho.symm
    subst out
    exact ⟨.some r,rfl,by simpa [Action.pure,hstate] using hs⟩

def finish (lib : APIHost api neg inputs Metadata) (c : SavedContext) (p : Data) :
    Action (Host.world (Metadata:=Metadata) api inputs) Gradients :=
  (call .returnTuple (lib.view p.dx c.inputShape).call).bind fun dx =>
  (transferView lib p.dw c.weightShape c.weightDevice).bind fun dw =>
  (optionalReturn lib p.db c.biasShape c.biasDevice).bind fun db =>
  Action.pure (dx.tensor,dw,db,none)

structure Route (lib : APIHost api neg inputs Metadata) (c : SavedContext) (p : Data)
    (initial : State Store Metadata) where
  dx : Dense
  dxOK : ((lib.view p.dx c.inputShape).call.run initial).2=.ok dx
  wr : TransferRoute lib p.dw c.weightShape c.weightDevice ((lib.view p.dx c.inputShape).call.run initial).1
  br : OptionalReturn lib c.biasShape c.biasDevice wr.final p.db

variable {c : SavedContext} {p : Data}

def Route.output (r : Route lib c p initial) : Gradients := (r.dx.tensor,r.wr.output,r.br.output,none)
def Route.final (r : Route lib c p initial) := r.br.final
def Route.transition (r : Route lib c p initial) : Transition api neg inputs initial r.final :=
  ((lib.view p.dx c.inputShape).transition initial).then (r.wr.transition.then r.br.transition)

theorem finish_route (out : Gradients)
    (ok : ((finish lib c p).run initial).result=.ok out) :
    ∃ r : Route lib c p initial,r.output=out ∧ ((finish lib c p).run initial).state=r.final := by
  obtain ⟨dx,hdx,hw,hsx,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨dw,hdw,hb,hsw,_⟩ := bind_success _ _ _ _ hw
  obtain ⟨wr,hew,hwstate⟩ := transfer_route dw hdw
  subst dw
  rw [hwstate] at hb hsw
  obtain ⟨db,hdb,hret,hsb,_⟩ := bind_success _ _ _ _ hb
  obtain ⟨br,heb,hbstate⟩ := optional_route p.db c.biasShape c.biasDevice db hdb
  subst db
  let route : Route lib c p initial := ⟨dx,(call_ok _ _ _ _).mp hdx,wr,br⟩
  have hout : out=route.output := by simpa [Action.pure,route,Route.output] using hret.symm
  subst out
  exact ⟨route,rfl,hsx.trans (hsw.trans (hsb.trans hbstate))⟩

#axiomsClean transfer_values
#axiomsClean transfer_real
#axiomsClean finish_route
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
