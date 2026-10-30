import bench.tritonbench_g.fast_layernorm.Proofs.Host
import bench.tritonbench_g.fast_layernorm.Proofs.Compiler

/-! Actual materialization and empty_like/statistics-allocation sequence.
Successful local calls are extracted from the executable action; no prepared
snapshot, fresh output or whole-wrapper success is assumed as an API field. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Preparation
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type}

def materialize (lib : APIHost api neg inputs Metadata) (owner : Nat)
    (t : Storage.Tensor) (shape : List Nat) : Action (Host.world (Metadata:=Metadata) api inputs) Dense :=
  (denseOn lib owner t).bind fun dense => call .contiguous (lib.view dense shape).call

structure MaterializeRoute (lib : APIHost api neg inputs Metadata) (owner : Nat)
    (t : Storage.Tensor) (shape : List Nat) (initial : State Store Metadata) where
  dense : DenseRoute lib owner t initial
  output : Dense
  viewOK : ((lib.view dense.output shape).call.run dense.final).2=.ok output

variable {lib : APIHost api neg inputs Metadata} {owner : Nat} {t : Storage.Tensor}
  {shape : List Nat} {initial : State Store Metadata}

def MaterializeRoute.final (r : MaterializeRoute lib owner t shape initial) :=
  ((lib.view r.dense.output shape).call.run r.dense.final).1

def MaterializeRoute.transition (r : MaterializeRoute lib owner t shape initial) :
    Transition api neg inputs initial r.final := r.dense.transition.then
      ((lib.view r.dense.output shape).transition r.dense.final)

theorem materialize_route (out : Dense)
    (ok : ((materialize lib owner t shape).run initial).result=.ok out) :
    ∃ r : MaterializeRoute lib owner t shape initial,r.output=out ∧
      ((materialize lib owner t shape).run initial).state=r.final := by
  obtain ⟨d,hd,hv,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨dr,he,hstate⟩ := dense_route d hd
  subst d
  have hok : ((lib.view dr.output shape).call.run dr.final).2=.ok out := by
    apply (call_ok _ _ _ _).mp
    simpa only [hstate] using hv
  refine ⟨⟨dr,out,hok⟩,rfl,?_⟩
  simpa only [hstate] using hs

theorem materialize_metadata (r : MaterializeRoute lib owner t shape initial) :
    r.output.tensor.view.shape=shape ∧ r.output.count=t.view.shape.prod ∧
    r.output.owner=owner ∧ r.final.live r.output.tensor := by
  have hv := lib.viewed r.dense.output shape r.dense.final r.output r.viewOK
  have hd := dense_metadata r.dense
  refine ⟨hv.shape,?_,?_,hv.present⟩
  · rw [hv.count,r.dense.output.countEq,hd.1]
  · change r.output.tensor.backing.owner=owner
    rw [hv.backing,←r.dense.output.tensor.owner,hd.2.1]

theorem materialize_values (r : MaterializeRoute lib owner t shape initial)
    (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.logical api neg r.final.memory r.output.tensor i=Storage.logical api neg initial.memory t i := by
  have hv := lib.viewed r.dense.output shape r.dense.final r.output r.viewOK
  have hd := dense_metadata r.dense
  have hcount : i<r.dense.output.count := by rw [r.dense.output.countEq,hd.1]; exact hi
  exact (hv.values i hcount).trans (dense_values r.dense i hi)

theorem materialize_real (r : MaterializeRoute lib owner t shape initial)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<t.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) r.output.tensor i=Storage.realLogical m t i := by
  have hv := lib.viewed r.dense.output shape r.dense.final r.output r.viewOK
  have hd := dense_metadata r.dense
  have hcount : i<r.dense.output.count := by rw [r.dense.output.countEq,hd.1]; exact hi
  exact (hv.realValues _ i hcount).trans (dense_real_values r.dense m i hi)

def optionalMaterialize (lib : APIHost api neg inputs Metadata) (owner : Nat)
    (t : Option Storage.Tensor) : Action (Host.world (Metadata:=Metadata) api inputs) (Option Dense) :=
  match t with
  | none => Action.pure none
  | some b => (materialize lib owner b [b.view.shape.prod]).bind fun d => Action.pure (some d)

inductive OptionalRoute (lib : APIHost api neg inputs Metadata) (owner : Nat)
    (initial : State Store Metadata) : Option Storage.Tensor → Type
  | none : OptionalRoute lib owner initial none
  | some {b : Storage.Tensor} (route : MaterializeRoute lib owner b [b.view.shape.prod] initial) :
      OptionalRoute lib owner initial (some b)

def OptionalRoute.output {b : Option Storage.Tensor} : OptionalRoute lib owner initial b → Option Dense
  | .none => Option.none | .some r => Option.some r.output

def OptionalRoute.final {b : Option Storage.Tensor} : OptionalRoute lib owner initial b → State Store Metadata
  | .none => initial | .some r => r.final

def OptionalRoute.transition {b : Option Storage.Tensor} (r : OptionalRoute lib owner initial b) :
    Transition api neg inputs initial r.final :=
  match r with | .none => .refl initial | .some r => r.transition

theorem optional_route (b : Option Storage.Tensor) (out : Option Dense)
    (ok : ((optionalMaterialize lib owner b).run initial).result=.ok out) :
    ∃ r : OptionalRoute lib owner initial b,r.output=out ∧
      ((optionalMaterialize lib owner b).run initial).state=r.final := by
  cases b with
  | none =>
    have h : out=none := by simpa [optionalMaterialize,Action.pure] using ok.symm
    subst out
    exact ⟨.none,rfl,rfl⟩
  | some b =>
    obtain ⟨d,hd,ho,hs,_⟩ := bind_success _ _ _ _ ok
    obtain ⟨r,he,hstate⟩ := materialize_route d hd
    subst d
    have h : out=some r.output := by simpa [Action.pure] using ho.symm
    subst out
    exact ⟨.some r,rfl,by simpa [Action.pure,hstate] using hs⟩

structure Data where
  (x w y r mu : Dense)
  bias : Option Dense

/-- Source order: X contiguous+view, W contiguous+view, optional bias contiguous+view,
Y=empty_like(X), r=empty(rows,fp32), mu=empty(rows,fp32). -/
def prepare (lib : APIHost api neg inputs Metadata) (x w : Storage.Tensor)
    (bias : Option Storage.Tensor) (rowCount N : Nat) :
    Action (Host.world (Metadata:=Metadata) api inputs) Data :=
  (materialize lib x.view.device x [rowCount,N]).bind fun xd =>
  (materialize lib x.view.device w [w.view.shape.prod]).bind fun wd =>
  (optionalMaterialize lib x.view.device bias).bind fun bd =>
  (call (.allocate .y) (lib.emptyLike xd).call).bind fun y =>
  (call (.allocate .rstd) (lib.emptyStats x.view.device rowCount).call).bind fun r =>
  (call (.allocate .mean) (lib.emptyStats x.view.device rowCount).call).bind fun mu =>
  Action.pure ⟨xd,wd,y,r,mu,bd⟩

structure Route (lib : APIHost api neg inputs Metadata) (x w : Storage.Tensor)
    (bias : Option Storage.Tensor) (rowCount N : Nat) (initial : State Store Metadata) where
  xr : MaterializeRoute lib x.view.device x [rowCount,N] initial
  wr : MaterializeRoute lib x.view.device w [w.view.shape.prod] xr.final
  br : OptionalRoute lib x.view.device wr.final bias
  (y r mu : Dense)
  yOK : ((lib.emptyLike xr.output).call.run br.final).2=.ok y
  rOK : ((lib.emptyStats x.view.device rowCount).call.run ((lib.emptyLike xr.output).call.run br.final).1).2=.ok r
  muOK : ((lib.emptyStats x.view.device rowCount).call.run
    ((lib.emptyStats x.view.device rowCount).call.run ((lib.emptyLike xr.output).call.run br.final).1).1).2=.ok mu

variable {x w : Storage.Tensor} {bias : Option Storage.Tensor} {rowCount N : Nat}

def Route.data (r : Route lib x w bias rowCount N initial) : Data :=
  ⟨r.xr.output,r.wr.output,r.y,r.r,r.mu,r.br.output⟩
def Route.sY (r : Route lib x w bias rowCount N initial) := ((lib.emptyLike r.xr.output).call.run r.br.final).1
def Route.sR (r : Route lib x w bias rowCount N initial) := ((lib.emptyStats x.view.device rowCount).call.run r.sY).1
def Route.final (r : Route lib x w bias rowCount N initial) := ((lib.emptyStats x.view.device rowCount).call.run r.sR).1

def Route.tailBias (r : Route lib x w bias rowCount N initial) : Transition api neg inputs r.br.final r.final :=
  ((lib.emptyLike r.xr.output).transition r.br.final).then
    (((lib.emptyStats x.view.device rowCount).transition r.sY).then ((lib.emptyStats x.view.device rowCount).transition r.sR))
def Route.tailWeight (r : Route lib x w bias rowCount N initial) : Transition api neg inputs r.wr.final r.final :=
  r.br.transition.then r.tailBias
def Route.tailX (r : Route lib x w bias rowCount N initial) : Transition api neg inputs r.xr.final r.final :=
  r.wr.transition.then r.tailWeight
def Route.transition (r : Route lib x w bias rowCount N initial) : Transition api neg inputs initial r.final :=
  r.xr.transition.then r.tailX

theorem prepare_route (out : Data)
    (ok : ((prepare lib x w bias rowCount N).run initial).result=.ok out) :
    ∃ r : Route lib x w bias rowCount N initial,r.data=out ∧
      ((prepare lib x w bias rowCount N).run initial).state=r.final := by
  obtain ⟨xd,hx,hw,hsx,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨xr,hex,hsxr⟩ := materialize_route xd hx
  subst xd
  rw [hsxr] at hw hsx
  obtain ⟨wd,hwm,hb,hsw,_⟩ := bind_success _ _ _ _ hw
  obtain ⟨wr,hew,hswr⟩ := materialize_route wd hwm
  subst wd
  rw [hswr] at hb hsw
  obtain ⟨bd,hbm,hy,hsb,_⟩ := bind_success _ _ _ _ hb
  obtain ⟨br,heb,hsbr⟩ := optional_route bias bd hbm
  subst bd
  rw [hsbr] at hy hsb
  obtain ⟨y,hym,hr,hsy,_⟩ := bind_success _ _ _ _ hy
  obtain ⟨r,hrm,hmu,hsr,_⟩ := bind_success _ _ _ _ hr
  obtain ⟨mu,hmum,hret,hsmu,_⟩ := bind_success _ _ _ _ hmu
  have hyok := (call_ok _ _ _ _).mp hym
  have hrok := (call_ok _ _ _ _).mp hrm
  have hmuok := (call_ok _ _ _ _).mp hmum
  let route : Route lib x w bias rowCount N initial := ⟨xr,wr,br,y,r,mu,hyok,hrok,hmuok⟩
  have hout : out=route.data := by simpa [Action.pure,route,Route.data] using hret.symm
  subst out
  refine ⟨route,rfl,?_⟩
  exact hsx.trans (hsw.trans (hsb.trans (hsy.trans (hsr.trans hsmu))))

theorem prepare_x_values (r : Route lib x w bias rowCount N initial) (i : Nat) (hi : i<x.view.shape.prod) :
    Storage.logical api neg r.final.memory r.data.x.tensor i=Storage.logical api neg initial.memory x i := by
  have hm := materialize_metadata r.xr
  have hv : i<r.xr.output.tensor.view.shape.prod := by rw [←r.xr.output.countEq,hm.2.1]; exact hi
  exact (r.tailX.values _ hm.2.2.2 i hv).trans (materialize_values r.xr i hi)

theorem prepare_w_values (r : Route lib x w bias rowCount N initial)
    (present : initial.live w) (i : Nat) (hi : i<w.view.shape.prod) :
    Storage.logical api neg r.final.memory r.data.w.tensor i=Storage.logical api neg initial.memory w i := by
  have hm := materialize_metadata r.wr
  have hv : i<r.wr.output.tensor.view.shape.prod := by rw [←r.wr.output.countEq,hm.2.1]; exact hi
  exact (r.tailWeight.values _ hm.2.2.2 i hv).trans
    ((materialize_values r.wr i hi).trans (r.xr.transition.values w present i hi))

theorem prepare_x_real (r : Route lib x w bias rowCount N initial)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<x.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) r.data.x.tensor i=Storage.realLogical m x i := by
  have hm := materialize_metadata r.xr
  have hv : i<r.xr.output.tensor.view.shape.prod := by rw [←r.xr.output.countEq,hm.2.1]; exact hi
  exact (r.tailX.realValues _ _ hm.2.2.2 i hv).trans (materialize_real r.xr m i hi)

theorem prepare_w_real (r : Route lib x w bias rowCount N initial) (present : initial.live w)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<w.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) r.data.w.tensor i=Storage.realLogical m w i := by
  have hm := materialize_metadata r.wr
  have hv : i<r.wr.output.tensor.view.shape.prod := by rw [←r.wr.output.countEq,hm.2.1]; exact hi
  exact (r.tailWeight.realValues _ _ hm.2.2.2 i hv).trans
    ((materialize_real r.wr _ i hi).trans (r.xr.transition.realValues m w present i hi))

theorem materialize_owner (r : MaterializeRoute lib owner t shape initial) : r.final.owner=initial.owner :=
  (lib.viewOwner r.dense.output shape r.dense.final).trans (dense_owner r.dense)

theorem optional_owner (r : OptionalRoute lib owner initial bias) : r.final.owner=initial.owner := by
  cases r with
  | none => rfl
  | some r => exact materialize_owner r

theorem prepare_owner (r : Route lib x w bias rowCount N initial) : r.final.owner=initial.owner :=
  (lib.emptyStatsOwner x.view.device rowCount r.sR).trans
    ((lib.emptyStatsOwner x.view.device rowCount r.sY).trans
      ((lib.emptyLikeOwner r.xr.output r.br.final).trans
        ((optional_owner r.br).trans ((materialize_owner r.wr).trans (materialize_owner r.xr)))))

theorem prepare_bias_values (r : Route lib x w bias rowCount N initial)
    (original : Storage.Tensor) (dense : Dense) (hb : bias=some original) (hd : r.data.bias=some dense)
    (present : initial.live original) (i : Nat) (hi : i<original.view.shape.prod) :
    Storage.logical api neg r.final.memory dense.tensor i=Storage.logical api neg initial.memory original i := by
  subst bias
  cases hbr : r.br with
  | some br =>
    have he : dense=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hd.symm
    subst dense
    have hm := materialize_metadata br
    have hv : i<br.output.tensor.view.shape.prod := by rw [←br.output.countEq,hm.2.1]; exact hi
    have ht := (r.xr.transition.then r.wr.transition).values original present i hi
    have hbval := materialize_values br i hi
    have tail := r.tailBias.values br.output.tensor (by simpa [hbr,OptionalRoute.final] using hm.2.2.2) i hv
    have hbf : r.br.final=br.final := congrArg OptionalRoute.final hbr
    rw [hbf] at tail
    exact tail.trans (hbval.trans ht)

theorem prepare_bias_real (r : Route lib x w bias rowCount N initial)
    (original : Storage.Tensor) (dense : Dense) (hb : bias=some original) (hd : r.data.bias=some dense)
    (present : initial.live original) (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<original.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) dense.tensor i=Storage.realLogical m original i := by
  subst bias
  cases hbr : r.br with
  | some br =>
    have he : dense=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hd.symm
    subst dense
    have hm := materialize_metadata br
    have hv : i<br.output.tensor.view.shape.prod := by rw [←br.output.countEq,hm.2.1]; exact hi
    have ht := (r.xr.transition.then r.wr.transition).realValues m original present i hi
    have hbval := materialize_real br ((r.xr.transition.then r.wr.transition).shadow m) i hi
    have tail := r.tailBias.realValues (br.transition.shadow ((r.xr.transition.then r.wr.transition).shadow m))
      br.output.tensor (by simpa [hbr,OptionalRoute.final] using hm.2.2.2) i hv
    have hbshadow : r.br.transition.shadow=br.transition.shadow := by rw [hbr]; rfl
    change Storage.realLogical (r.tailBias.shadow (r.br.transition.shadow
      (r.wr.transition.shadow (r.xr.transition.shadow m)))) br.output.tensor i=_
    rw [hbshadow]
    exact tail.trans (hbval.trans ht)

#axiomsClean prepare_owner
#axiomsClean prepare_bias_values
#axiomsClean prepare_bias_real
#axiomsClean materialize_route
#axiomsClean materialize_values
#axiomsClean materialize_real
#axiomsClean prepare_route
#axiomsClean prepare_x_values
#axiomsClean prepare_w_values
#axiomsClean prepare_x_real
#axiomsClean prepare_w_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Preparation
