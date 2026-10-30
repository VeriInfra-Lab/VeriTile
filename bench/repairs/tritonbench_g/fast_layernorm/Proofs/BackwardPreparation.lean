import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Preparation

/-! Actual backward allocations: materialize dY, then empty_like of saved X,
W, and optional bias. The saved statistics are handles, never assumed formulas. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardPreparation
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type}

def savedInputs (c : SavedContext) : List Storage.Tensor :=
  ([c.x,c.w,c.r,c.mu]++c.bias.toList).map Dense.tensor

structure SavedLayout (c : SavedContext) : Prop where
  positive : 0<c.columns
  xShape : c.x.tensor.view.shape=[c.rows,c.columns]
  xCount : c.x.count=c.rows*c.columns
  rCount : c.r.count=c.rows
  muCount : c.mu.count=c.rows
  wCount : c.columns≤c.w.count
  biasCount : ∀ b,c.bias=some b → c.columns≤b.count
  owners : ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → t.owner=c.x.owner
  inputShape : c.inputShape.prod=c.x.count
  weightShape : c.weightShape.prod=c.w.count
  biasShape : ∀ b,c.bias=some b → ∃ shape,c.biasShape=some shape ∧ shape.prod=b.count
  biasDevice : c.bias.isSome=c.biasDevice.isSome
  weightOwner : c.weightDevice=c.w.owner
  biasOwner : ∀ b,c.bias=some b → c.biasDevice.getD 0=b.owner
  settings : Compiler.settings c.columns=.ok (c.block,c.warps)

structure Data where
  (dy dx dw : Dense)
  db : Option Dense

def optionalAllocate (lib : APIHost api neg inputs Metadata) (b : Option Dense) :
    Action (Host.world (Metadata:=Metadata) api inputs) (Option Dense) :=
  match b with
  | none => Action.pure none
  | some b => (call (.allocate .bias) (lib.emptyLike b).call).bind fun db => Action.pure (some db)

inductive OptionalAllocation (lib : APIHost api neg inputs Metadata)
    (initial : State Store Metadata) : Option Dense → Type
  | none : OptionalAllocation lib initial none
  | some {b : Dense} (output : Dense) (ok : ((lib.emptyLike b).call.run initial).2=.ok output) :
      OptionalAllocation lib initial (some b)

variable {lib : APIHost api neg inputs Metadata} {initial : State Store Metadata}

def OptionalAllocation.output {b : Option Dense} : OptionalAllocation lib initial b → Option Dense
  | .none => Option.none | .some out _ => Option.some out

def OptionalAllocation.final {b : Option Dense} (_ : OptionalAllocation lib initial b) : State Store Metadata :=
  match b with | Option.none => initial | Option.some b => ((lib.emptyLike b).call.run initial).1

def OptionalAllocation.transition {b : Option Dense} (r : OptionalAllocation lib initial b) :
    Transition api neg inputs initial r.final :=
  match b with | Option.none => .refl initial | Option.some b => (lib.emptyLike b).transition initial

theorem optional_route (b : Option Dense) (out : Option Dense)
    (ok : ((optionalAllocate lib b).run initial).result=.ok out) :
    ∃ r : OptionalAllocation lib initial b,r.output=out ∧
      ((optionalAllocate lib b).run initial).state=r.final := by
  cases b with
  | none =>
    have h : out=none := by simpa [optionalAllocate,Action.pure] using ok.symm
    subst out
    exact ⟨.none,rfl,rfl⟩
  | some b =>
    obtain ⟨d,hd,ho,hs,_⟩ := bind_success _ _ _ _ ok
    have hc := (call_ok _ _ _ _).mp hd
    have h : out=some d := by simpa [Action.pure] using ho.symm
    subst out
    exact ⟨.some d hc,rfl,hs⟩

def prepare (lib : APIHost api neg inputs Metadata) (c : SavedContext) (dy : Storage.Tensor) :
    Action (Host.world (Metadata:=Metadata) api inputs) Data :=
  (materialize lib c.x.tensor.view.device dy [c.rows,c.columns]).bind fun yd =>
  (call (.allocate .x) (lib.emptyLike c.x).call).bind fun dx =>
  (call (.allocate .weight) (lib.emptyLike c.w).call).bind fun dw =>
  (optionalAllocate lib c.bias).bind fun db =>
  Action.pure ⟨yd,dx,dw,db⟩

structure Route (lib : APIHost api neg inputs Metadata) (c : SavedContext) (dy : Storage.Tensor)
    (initial : State Store Metadata) where
  dyr : MaterializeRoute lib c.x.tensor.view.device dy [c.rows,c.columns] initial
  (dx dw : Dense)
  dxOK : ((lib.emptyLike c.x).call.run dyr.final).2=.ok dx
  dwOK : ((lib.emptyLike c.w).call.run ((lib.emptyLike c.x).call.run dyr.final).1).2=.ok dw
  db : OptionalAllocation lib ((lib.emptyLike c.w).call.run ((lib.emptyLike c.x).call.run dyr.final).1).1 c.bias

variable {c : SavedContext} {dy : Storage.Tensor}

def Route.sX (r : Route lib c dy initial) := ((lib.emptyLike c.x).call.run r.dyr.final).1
def Route.sW (r : Route lib c dy initial) := ((lib.emptyLike c.w).call.run r.sX).1
def Route.final (r : Route lib c dy initial) := r.db.final
def Route.data (r : Route lib c dy initial) : Data := ⟨r.dyr.output,r.dx,r.dw,r.db.output⟩
def Route.tail (r : Route lib c dy initial) : Transition api neg inputs r.dyr.final r.final :=
  ((lib.emptyLike c.x).transition r.dyr.final).then
    (((lib.emptyLike c.w).transition r.sX).then r.db.transition)
def Route.transition (r : Route lib c dy initial) : Transition api neg inputs initial r.final :=
  r.dyr.transition.then r.tail

theorem prepare_route (out : Data)
    (ok : ((prepare lib c dy).run initial).result=.ok out) :
    ∃ r : Route lib c dy initial,r.data=out ∧ ((prepare lib c dy).run initial).state=r.final := by
  obtain ⟨yd,hy,hx,hsy,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨yr,hey,hstate⟩ := materialize_route yd hy
  subst yd
  rw [hstate] at hx hsy
  obtain ⟨dx,hdx,hw,hsx,_⟩ := bind_success _ _ _ _ hx
  obtain ⟨dw,hdw,hb,hsw,_⟩ := bind_success _ _ _ _ hw
  obtain ⟨db,hdb,hret,hsb,_⟩ := bind_success _ _ _ _ hb
  obtain ⟨br,heb,hbstate⟩ := optional_route c.bias db hdb
  subst db
  let route : Route lib c dy initial := ⟨yr,dx,dw,(call_ok _ _ _ _).mp hdx,(call_ok _ _ _ _).mp hdw,br⟩
  have hout : out=route.data := by simpa [Action.pure,route,Route.data] using hret.symm
  subst out
  exact ⟨route,rfl,hsy.trans (hsx.trans (hsw.trans (hsb.trans hbstate)))⟩

theorem dy_values (r : Route lib c dy initial) (i : Nat) (hi : i<dy.view.shape.prod) :
    Storage.logical api neg r.final.memory r.data.dy.tensor i=Storage.logical api neg initial.memory dy i := by
  have hm := materialize_metadata r.dyr
  have hv : i<r.dyr.output.tensor.view.shape.prod := by rw [←r.dyr.output.countEq,hm.2.1]; exact hi
  exact (r.tail.values _ hm.2.2.2 i hv).trans (materialize_values r.dyr i hi)

theorem dy_real (r : Route lib c dy initial) (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<dy.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) r.data.dy.tensor i=Storage.realLogical m dy i := by
  have hm := materialize_metadata r.dyr
  have hv : i<r.dyr.output.tensor.view.shape.prod := by rw [←r.dyr.output.countEq,hm.2.1]; exact hi
  exact (r.tail.realValues _ _ hm.2.2.2 i hv).trans (materialize_real r.dyr m i hi)

#axiomsClean prepare_route
#axiomsClean dy_values
#axiomsClean dy_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardPreparation
