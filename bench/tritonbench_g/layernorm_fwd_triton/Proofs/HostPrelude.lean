import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Torch

namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace HostPrelude
open Torch
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
variable {α Store Byte Host : Type} {api : Memory.API α Store Byte}
  {neg : α → α} {caller : Nat → Prop}

/-- Four assertions execute before entering the device context. The role tags
identify the two ranks and feature/head equalities in source order. -/
def validate (x w : Tensor) : Action (Torch.world (Host:=Host) api caller) Shape :=
  (check (.assertShape .x) (x.shape.length=3) (.assertion (.assertShape .x))).bind fun _ =>
  (check (.assertShape .weight) (w.shape.length=2) (.assertion (.assertShape .weight))).bind fun _ =>
  (check (.shape .x) (x.shape.getLast?=w.shape.getLast?) (.assertion (.shape .x))).bind fun _ =>
  (check (.shape .weight) (x.shape[1]?=w.shape[0]?) (.assertion (.shape .weight))).bind fun _ =>
  Action.pure ⟨x.shape[0]?.getD 0,x.shape[1]?.getD 0,x.shape[2]?.getD 0⟩

theorem validate_shapes (x w : Tensor) (s : State Store Host) (shape : Shape)
    (ok : ((validate (api:=api) (caller:=caller) x w).run s).result=.ok shape) :
    x.shape=[shape.sequences,shape.heads,shape.columns] ∧ w.shape=[shape.heads,shape.columns] := by
  by_cases hx : x.shape.length=3
  · by_cases hw : w.shape.length=2
    · obtain ⟨a,b,c,hxs⟩ := List.length_eq_three.mp hx
      obtain ⟨d,e,hws⟩ := List.length_eq_two.mp hw
      by_cases hc : c=e
      · by_cases hb : b=d
        · simp [validate,Action.bind,check,Action.pure,hxs,hws,hc,hb] at ok
          cases ok
          simp [hxs,hws,hc,hb]
        · simp [validate,Action.bind,check,Action.pure,hxs,hws,hc,hb] at ok
      · simp [validate,Action.bind,check,Action.pure,hxs,hws,hc] at ok
    · simp [validate,Action.bind,check,hx,hw] at ok
  · simp [validate,Action.bind,check,hx] at ok

/-- The repaired wrapper passes X/W directly and calls only empty_like. -/
def prepare (lib : APIHost api neg caller Host) (x w : Tensor) :
    Action (Torch.world (Host:=Host) api caller) (Tensor × Tensor × Tensor) :=
  (call (.allocate .y) (lib.allocate x).call).bind fun y => Action.pure (x,w,y)

structure Route (lib : APIHost api neg caller Host) (x w : Tensor) (initial : State Store Host) where
  y : Tensor
  allocateY : ((lib.allocate x).call.run initial).2=.ok y

variable {lib : APIHost api neg caller Host} {x w : Tensor} {initial : State Store Host}

def Route.xr (_ : Route lib x w initial) : Tensor := x
def Route.wp (_ : Route lib x w initial) : Tensor := w
def Route.finish (_ : Route lib x w initial) : State Store Host := ((lib.allocate x).call.run initial).1
def Route.shadow (_ : Route lib x w initial) : (Nat → ℝ) → (Nat → ℝ) :=
  ((lib.allocate x).shadow initial).run

theorem call_ok (effect : Local (Torch.world (Host:=Host) api caller) Value) (stage : Stage)
    (s : State Store Host) (v : Value) (h : ((call stage effect).run s).result=.ok v) :
    (effect.run s).2=.ok v := by
  cases he : (effect.run s).2 <;> simp [call,he,Except.mapError] at h ⊢
  exact h

theorem prepare_route (lib : APIHost api neg caller Host) (x w : Tensor)
    (initial : State Store Host) (value : Tensor × Tensor × Tensor)
    (ok : ((prepare lib x w).run initial).result=.ok value) :
    ∃ r : Route lib x w initial, value=(r.xr,r.wp,r.y) ∧
      ((prepare lib x w).run initial).state=r.finish := by
  unfold prepare at ok ⊢
  obtain ⟨y,hy,hv,hs,_⟩ := bind_success _ _ _ _ ok
  have hv' : (x,w,y)=value := by simpa [Action.pure] using hv
  exact ⟨⟨y,call_ok _ _ _ _ hy⟩,hv'.symm,hs⟩

theorem route_inputs (lib : APIHost api neg caller Host) (x w : Tensor)
    (initial : State Store Host) (r : Route lib x w initial)
    (xp : initial.live x) (wp : initial.live w)
    (xn : x.negative=Bool.false) (wn : w.negative=Bool.false)
    (xr : initial.ready x (initial.streams x.device))
    (wr : initial.ready w (initial.streams w.device)) :
    Materialized api neg x initial r.finish r.shadow r.xr ∧
    Materialized api neg w initial r.finish r.shadow r.wp := by
  let effect := lib.allocate x
  have each (t : Tensor) (hp : initial.live t) (hn : t.negative=Bool.false)
      (hr : initial.ready t (initial.streams t.device)) :
      Materialized api neg t initial r.finish r.shadow t := by
    refine ⟨rfl,rfl,hn,(effect.preserves initial).live t hp,?_,?_,?_⟩
    · exact (effect.preserves initial).values t hp
    · intro m index hi
      exact (effect.shadow initial).values m t hp index hi
    · change (effect.call.run initial).1.ready t ((effect.call.run initial).1.streams t.device)
      rw [(effect.preserves initial).streams]
      exact (effect.preserves initial).ready t _ hr
  exact ⟨each x xp xn xr,each w wp wn wr⟩

theorem route_owner (lib : APIHost api neg caller Host) (x w : Tensor)
    (initial : State Store Host) (r : Route lib x w initial) : r.finish.owner=initial.owner :=
  lib.allocate_owner x initial

theorem route_shadow_frame (lib : APIHost api neg caller Host) (x w : Tensor)
    (initial : State Store Host) (r : Route lib x w initial)
    (m : Nat → ℝ) (i : Nat) (hi : caller i) : r.shadow m i=m i :=
  ((lib.allocate x).shadow initial).frame m i hi

#axiomsClean validate_shapes
#axiomsClean prepare_route
#axiomsClean route_inputs
#axiomsClean route_owner
#axiomsClean route_shadow_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end HostPrelude
end VeriTile.Bench.Repairs.LayernormFwdTriton
