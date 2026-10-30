import bench.tritonbench_g.layernorm_fwd_triton.Proofs.HostLayout

namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace HostPrepared
open Torch Address Memory Execution HostPrelude HostLayout
noncomputable section
open scoped Classical

/-- Immutable actual launch descriptors. Every proof field below is derived
from the single empty_like call and admitted input facts in `ofRoute`. -/
structure Data (shape : Shape) (caller : Nat → Prop) where
  (x w y : Tensor)
  xv : View3 x shape
  wv : View2 w shape
  yv : View3 y shape
  (x4 : x.allocation.width=4) (w4 : w.allocation.width=4) (y4 : y.allocation.width=4)
  (xPositive : x.negative=Bool.false) (wPositive : w.negative=Bool.false) (yPositive : y.negative=Bool.false)
  (wDevice : w.device=x.device) (yDevice : y.device=x.device)
  injective : ∀ i j, Torch.Coordinate y i → Torch.Coordinate y j → address y i=address y j → i=j
  freshX : ∀ i, Covers x i → ¬Covers y i
  freshW : ∀ i, Covers w i → ¬Covers y i
  freshCaller : ∀ i, Covers y i → ¬caller i

def Data.strides (p : Data shape caller) : Strides := HostLayout.strides p.xv p.wv p.yv
def Data.layout (p : Data shape caller) : Layout shape p.strides :=
  HostLayout.layout p.xv p.wv p.yv p.x4 p.w4 p.y4 p.injective p.freshX p.freshW

def Ready (p : Data shape caller) (s : State Store Host) : Prop :=
  s.owner=p.x.device ∧ s.live p.x ∧ s.live p.w ∧ s.live p.y ∧
  s.ready p.x (s.streams p.x.device) ∧ s.ready p.w (s.streams p.x.device) ∧ s.ready p.y (s.streams p.x.device)

variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}

def ofRoute (lib : APIHost api neg caller Host) (x w : Tensor) (initial : State Store Host)
    (r : Route lib x w initial) (shape : Shape)
    (xShape : x.shape=[shape.sequences,shape.heads,shape.columns]) (wShape : w.shape=[shape.heads,shape.columns])
    (x4 : x.allocation.width=4) (w4 : w.allocation.width=4)
    (xp : initial.live x) (wp : initial.live w)
    (xn : x.negative=Bool.false) (wn : w.negative=Bool.false) (wd : w.device=x.device)
    (xr : initial.ready x (initial.streams x.device)) (wr : initial.ready w (initial.streams w.device)) :
    Data shape caller := by
  have ym := lib.allocate_metadata x initial r.y r.allocateY
  exact {
    x := x, w := w, y := r.y
    xv := view3 x shape xShape
    wv := view2 w shape wShape
    yv := view3 r.y shape (ym.1.trans xShape)
    x4 := x4, w4 := w4, y4 := ym.2.1.trans x4
    xPositive := xn, wPositive := wn, yPositive := ym.2.2.2.1
    wDevice := wd, yDevice := ym.2.2.1
    injective := lib.allocate_injective x initial r.y r.allocateY
    freshX := fun i hx hy => (lib.allocate_fresh x initial r.y r.allocateY i hy).2 x xp hx
    freshW := fun i hw hy => (lib.allocate_fresh x initial r.y r.allocateY i hy).2 w wp hw
    freshCaller := fun i hy => (lib.allocate_fresh x initial r.y r.allocateY i hy).1 }

theorem ofRoute_ready (lib : APIHost api neg caller Host) (x w : Tensor) (initial : State Store Host)
    (r : Route lib x w initial) (shape : Shape)
    (xs ws x4 w4 xp wp xn wn wd xr wr) (owner : initial.owner=x.device) :
    Ready (ofRoute lib x w initial r shape xs ws x4 w4 xp wp xn wn wd xr wr) r.finish := by
  obtain ⟨mx,mw⟩ := route_inputs lib x w initial r xp wp xn wn xr wr
  obtain ⟨_,_,yd,_,yp⟩ := lib.allocate_metadata x initial r.y r.allocateY
  change r.finish.owner=x.device ∧ r.finish.live x ∧ r.finish.live w ∧ r.finish.live r.y ∧
    r.finish.ready x (r.finish.streams x.device) ∧
    r.finish.ready w (r.finish.streams x.device) ∧ r.finish.ready r.y (r.finish.streams x.device)
  refine ⟨(route_owner lib x w initial r).trans owner,mx.present,mw.present,yp,mx.ready,?_,?_⟩
  · simpa only [Route.wp,wd] using mw.ready
  · have hy := lib.allocate_ready x initial r.y r.allocateY
    have streams : r.finish.streams=initial.streams := (lib.allocate x).preserves initial |>.streams
    simpa only [streams,yd] using hy

theorem x_address (p : Data shape caller) (row : Row shape) (col : Fin shape.columns) :
    Address.xAddress p.layout row col=Torch.address p.x (index3 row col) :=
  HostLayout.x_address ..
theorem w_address (p : Data shape caller) (row : Row shape) (col : Fin shape.columns) :
    Address.wAddress p.layout row col=Torch.address p.w (index2 row col) :=
  HostLayout.w_address ..
theorem y_address (p : Data shape caller) (row : Row shape) (col : Fin shape.columns) :
    Address.yAddress p.layout row col=Torch.address p.y (index3 row col) :=
  HostLayout.y_address ..
theorem caller_outside (p : Data shape caller) (i : Nat) (hi : caller i)
    (row : Row shape) (col : Fin shape.columns) :
    i < Address.yAddress p.layout row col ∨ Address.yAddress p.layout row col+4 ≤ i :=
  HostLayout.caller_outside p.layout caller (fun j h1 h2 => p.freshCaller j ⟨h1,h2⟩) i hi row col

#axiomsClean ofRoute
#axiomsClean ofRoute_ready
#axiomsClean caller_outside
#guard_msgs (drop info) in
#auditModuleAxioms
end
end HostPrepared
end VeriTile.Bench.Repairs.LayernormFwdTriton
