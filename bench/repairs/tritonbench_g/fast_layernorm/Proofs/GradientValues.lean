import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientReturn

namespace VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {initial : State Store Metadata}

variable {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat}

theorem optional_values (r : OptionalReturn lib shape device initial b)
    (source : Dense) (out : Storage.Tensor) (hs : b=some source) (ho : r.output=some out)
    (i : Nat) (hi : i<source.count) :
    Storage.logical api neg r.final.memory out i=Storage.logical api neg initial.memory source.tensor i := by
  cases r with
  | none => contradiction
  | @some original tr =>
    have he : original=source := Option.some.inj hs
    subst source
    have heo : tr.output=out := by simpa [OptionalReturn.output] using ho
    subst out
    exact transfer_values tr i hi

theorem optional_real (r : OptionalReturn lib shape device initial b)
    (source : Dense) (out : Storage.Tensor) (hs : b=some source) (ho : r.output=some out)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<source.count) :
    Storage.realLogical (r.transition.shadow m) out i=Storage.realLogical m source.tensor i := by
  cases r with
  | none => contradiction
  | @some original tr =>
    have he : original=source := Option.some.inj hs
    subst source
    have heo : tr.output=out := by simpa [OptionalReturn.output] using ho
    subst out
    exact transfer_real tr m i hi

variable {c : SavedContext} {p : Data}

def Route.afterDX (r : Route lib c p initial) :
    Transition api neg inputs ((lib.view p.dx c.inputShape).call.run initial).1 r.final :=
  r.wr.transition.then r.br.transition

theorem dx_values (r : Route lib c p initial) (i : Nat) (hi : i<p.dx.count) :
    Storage.logical api neg r.final.memory r.output.1 i=Storage.logical api neg initial.memory p.dx.tensor i := by
  have hv := lib.viewed p.dx c.inputShape initial r.dx r.dxOK
  have hi' : i<r.dx.tensor.view.shape.prod := by rw [←r.dx.countEq,hv.count]; exact hi
  exact (r.afterDX.values _ hv.present i hi').trans (hv.values i hi)

theorem dx_real (r : Route lib c p initial) (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<p.dx.count) :
    Storage.realLogical (r.transition.shadow m) r.output.1 i=Storage.realLogical m p.dx.tensor i := by
  have hv := lib.viewed p.dx c.inputShape initial r.dx r.dxOK
  have hi' : i<r.dx.tensor.view.shape.prod := by rw [←r.dx.countEq,hv.count]; exact hi
  exact (r.afterDX.realValues _ _ hv.present i hi').trans (hv.realValues m i hi)

theorem dw_values (r : Route lib c p initial) (present : initial.live p.dw.tensor) (i : Nat) (hi : i<p.dw.count) :
    Storage.logical api neg r.final.memory r.output.2.1 i=Storage.logical api neg initial.memory p.dw.tensor i := by
  have hp := (transfer_metadata r.wr).2.2
  have ht := transfer_values r.wr i hi
  have hv := (lib.view p.dx c.inputShape).transition initial |>.values _ present i
    (by rw [←p.dw.countEq]; exact hi)
  have hm := lib.viewed p.dw c.weightShape _ r.wr.viewed r.wr.viewOK
  have hcount : i<r.wr.output.view.shape.prod := by
    rw [(transfer_metadata r.wr).1,←hm.shape,←r.wr.viewed.countEq,hm.count]
    exact hi
  exact (r.br.transition.values _ hp i hcount).trans (ht.trans hv)

theorem dw_real (r : Route lib c p initial) (present : initial.live p.dw.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<p.dw.count) :
    Storage.realLogical (r.transition.shadow m) r.output.2.1 i=Storage.realLogical m p.dw.tensor i := by
  have hp := (transfer_metadata r.wr).2.2
  have ht := transfer_real r.wr (((lib.view p.dx c.inputShape).transition initial).shadow m) i hi
  have hv := (lib.view p.dx c.inputShape).transition initial |>.realValues m _ present i
    (by rw [←p.dw.countEq]; exact hi)
  have hm := lib.viewed p.dw c.weightShape _ r.wr.viewed r.wr.viewOK
  have hcount : i<r.wr.output.view.shape.prod := by
    rw [(transfer_metadata r.wr).1,←hm.shape,←r.wr.viewed.countEq,hm.count]
    exact hi
  exact (r.br.transition.realValues _ _ hp i hcount).trans (ht.trans hv)

theorem db_values (r : Route lib c p initial) (source : Dense) (out : Storage.Tensor)
    (hs : p.db=some source) (ho : r.output.2.2.1=some out)
    (present : initial.live source.tensor) (i : Nat) (hi : i<source.count) :
    Storage.logical api neg r.final.memory out i=Storage.logical api neg initial.memory source.tensor i := by
  have ht := ((lib.view p.dx c.inputShape).transition initial).then r.wr.transition
  exact (optional_values r.br source out hs ho i hi).trans
    (ht.values _ present i (by rw [←source.countEq]; exact hi))

theorem db_real (r : Route lib c p initial) (source : Dense) (out : Storage.Tensor)
    (hs : p.db=some source) (ho : r.output.2.2.1=some out) (present : initial.live source.tensor)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<source.count) :
    Storage.realLogical (r.transition.shadow m) out i=Storage.realLogical m source.tensor i := by
  let ht := ((lib.view p.dx c.inputShape).transition initial).then r.wr.transition
  exact (optional_real r.br source out hs ho (ht.shadow m) i hi).trans
    (ht.realValues m _ present i (by rw [←source.countEq]; exact hi))

#axiomsClean dx_values
#axiomsClean dx_real
#axiomsClean dw_values
#axiomsClean dw_real
#axiomsClean db_values
#axiomsClean db_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
