import bench.tritonbench_g.fast_rms_layernorm.Proofs.Settings
import bench.tritonbench_g.fast_rms_layernorm.Proofs.StorageImage

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Host
open VeriTile.Triton Storage
noncomputable section

inductive InputShape
  | scalar
  | axes (leading : List Nat) (last : Nat)
  deriving DecidableEq

def InputShape.dims : InputShape → List Nat
  | .scalar => []
  | .axes leading last => leading++[last]

theorem all_shapes (dims : List Nat) : ∃ shape : InputShape,shape.dims=dims := by
  induction dims using List.reverseRecOn with
  | nil => exact ⟨.scalar,rfl⟩
  | append_singleton init last _ => exact ⟨.axes init last,rfl⟩

inductive Failure
  | scalar | emptyFeature | fusedBlock | device | reshape
  | resolveNeg | copy | allocation | setup | launch | zeroFill | internal | savedVersion
  deriving DecidableEq

/-- Values are logical coordinates at wrapper entry. Checked contiguous X/W
views preserve their original storage and values; only the upstream G is copied
into a fresh dX buffer because the input-gradient source overwrites that buffer. -/
structure Request (α : Type) where
  shape : InputShape
  weightShape : List Nat
  inputDevice : Nat
  weightDevice : Nat
  x : Nat → α
  w : Nat → α
  gemma : Bool
  eps : ℝ

def validate {α : Type} (r : Request α) : Except Failure Call :=
  match r.shape with
  | .scalar => .error .scalar
  | .axes leading cols =>
    if positive : 0<cols then
      if supported : Settings.block cols≤65536 then
        if r.inputDevice=r.weightDevice then
          if r.weightShape.prod=cols then
            .ok ⟨leading.prod,cols,Settings.block cols,positive,Settings.covered cols,supported,r.gemma,r.eps⟩
          else .error .reshape
        else .error .device
      else .error .fusedBlock
    else .error .emptyFeature

/-- Internal source shape/device preflight, after the public checked contract. -/
def preflight {α : Type} (r : Request α) : Except Failure Call :=
  match r.shape with
  | .scalar => .error .scalar
  | .axes leading cols =>
    if positive : 0<cols then
      if supported : Settings.block cols≤65536 then
        if r.inputDevice=r.weightDevice then
          .ok ⟨leading.prod,cols,Settings.block cols,positive,Settings.covered cols,supported,r.gemma,r.eps⟩
        else .error .device
      else .error .fusedBlock
    else .error .emptyFeature

theorem preflight_of_valid {α : Type} (r : Request α) (h : Call) (ok : validate r=.ok h) :
    preflight r=.ok h := by
  unfold validate at ok
  split at ok
  · cases ok
  · rename_i leading cols shape
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    simp_all [preflight]

theorem validated {α : Type} (r : Request α) (h : Call) (ok : validate r=.ok h) :
    ∃ leading,r.shape=.axes leading h.cols ∧ h.rows=leading.prod ∧
      h.block=Settings.block h.cols ∧ r.weightShape.prod=h.cols ∧
      r.inputDevice=r.weightDevice ∧ h.gemma=r.gemma ∧ h.eps=r.eps ∧
      r.shape.dims.prod=h.cells := by
  unfold validate at ok
  split at ok
  · cases ok
  · rename_i leading cols shape
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    split at ok <;> try cases ok
    refine ⟨leading,shape,rfl,rfl,by assumption,by assumption,rfl,rfl,?_⟩
    simp [shape,InputShape.dims,Call.cells]

theorem scalar_rejected {α : Type} (r : Request α) (shape : r.shape=.scalar) :
    validate r=.error .scalar := by simp [validate,shape]

theorem zero_rejected {α : Type} (r : Request α) (leading : List Nat) (shape : r.shape=.axes leading 0) :
    validate r=.error .emptyFeature := by simp [validate,shape]

abbrev CellHeap (α : Type) := RegionName × Nat → α

/-- All caller-visible backing bytes and storage identities at this call's
entry. Identities matter even for zero-byte tensors sharing a null pointer. -/
structure Caller where
  bytes : Nat → Prop
  identities : Nat → Prop

instance : CoeFun Caller (fun _ => Nat → Prop) := ⟨Caller.bytes⟩

def packed {α : Type} (r : Request α) (scratch : CellHeap α) : CellHeap α :=
  fun a => if a.1="x" then r.x a.2 else if a.1="w" then r.w a.2 else scratch a

@[simp] theorem packed_x {α : Type} (r : Request α) (scratch : CellHeap α) (i : Nat) :
    packed r scratch ("x",i)=r.x i := by simp [packed]
@[simp] theorem packed_w {α : Type} (r : Request α) (scratch : CellHeap α) (i : Nat) :
    packed r scratch ("w",i)=r.w i := by simp [packed]

/-- Only outputs are newly allocated. Forward X/W are ordinary direct views;
backward X/W/r reuse the saved tensors and may be caller-visible. -/
def newRole (p : Phase) (r : RegionName) : Prop := r∈outputs p

def NewFresh {h : Call} {p : Phase} (storage : Placement h p) (live : Caller) : Prop :=
  ∀ addr,live addr → ¬written storage addr

structure Prepared (h : Call) (p : Phase) (live : Caller) where
  storage : Placement h p
  fresh : NewFresh storage live
  freshIdentity : ∀ r hr,newRole p r → ¬live.identities (storage.allocation r hr).storageId
  freshBacking : ∀ r hr,newRole p r → ∀ addr,(storage.allocation r hr).buffer.base≤addr →
    addr<(storage.allocation r hr).buffer.base+(storage.allocation r hr).buffer.extent*4 → ¬live addr

abbrev Preparation (h : Call) (p : Phase) (live : Caller) := Except Failure (Prepared h p live)

/-- An ordinary preparation failure needs no placement, capacity or freshness
witness, even when successful allocations for this shape are impossible. -/
theorem preparation_failure (h : Call) (p : Phase) (live : Caller) (f : Failure) :
    ∃ result : Preparation h p live,result=.error f := ⟨.error f,rfl⟩

structure Context (α : Type) where
  call : Call
  inputShape : List Nat
  weightShape : List Nat
  device : Nat
  saved : CellHeap α
  storage : Placement call .forward

structure Upstream (α : Type) where
  shape : List Nat
  device : Nat
  values : Nat → α

def backwardValidate {α : Type} (ctx : Context α) (g : Upstream α) : Except Failure Unit :=
  if g.device=ctx.device then
    if g.shape.prod=ctx.call.cells then .ok () else .error .reshape
  else .error .device

def backwardDevice {α : Type} (ctx : Context α) (g : Upstream α) : Except Failure Unit :=
  if g.device=ctx.device then .ok () else .error .device

theorem backward_valid {α : Type} (ctx : Context α) (g : Upstream α)
    (ok : backwardValidate ctx g=.ok ()) : backwardDevice ctx g=.ok () ∧ g.shape.prod=ctx.call.cells := by
  unfold backwardValidate at ok
  split at ok <;> try cases ok
  split at ok <;> try cases ok
  simp_all [backwardDevice]

def seed {α : Type} (ctx : Context α) (g : Upstream α) : CellHeap α :=
  fun a => if a.1="dx" then g.values a.2 else ctx.saved a

def savedRole (r : RegionName) : Prop := r="x" ∨ r="w" ∨ r="r"
theorem saved_forward (r : RegionName) (hr : savedRole r) : r∈buffers .forward := by
  rcases hr with rfl|rfl|rfl <;> simp [buffers]
theorem saved_backward (r : RegionName) (hr : savedRole r) : r∈buffers .backward := by
  rcases hr with rfl|rfl|rfl <;> simp [buffers]

structure BackwardPrepared {α : Type} (ctx : Context α) (live : Caller) extends Prepared ctx.call .backward live where
  reuse : ∀ r (hr : savedRole r),
    storage.allocation r (saved_backward r hr)=
      ctx.storage.allocation r (saved_forward r hr)

theorem saved_addresses {α : Type} (ctx : Context α) (live : Caller)
    (p : BackwardPrepared ctx live) (r : RegionName) (hr : savedRole r) (off : Nat) (byte : Fin 4) :
    address p.storage r (saved_backward r hr) off byte=
      address ctx.storage r (saved_forward r hr) off byte := by
  unfold address
  rw [p.reuse r hr]

/-- Backward reads the saved values retrieved at call entry. Successful ordinary
saved-tensor/version retrieval binds these values to that entry storage; this
identity does not assert nonvisibility or restore prior raw bytes. -/
theorem seed_saved {α : Type} (ctx : Context α) (g : Upstream α)
    (r : RegionName) (hr : savedRole r) (off : Nat) : seed ctx g (r,off)=ctx.saved (r,off) := by
  rcases hr with rfl|rfl|rfl <;> simp [seed]

def zeroWeight {α : Type} (h : Call) (zero : α) (heap : CellHeap α) : CellHeap α :=
  fun a => if a.1="dw" ∧ a.2<h.cols then zero else heap a

@[simp] theorem zeroWeight_read {α : Type} (h : Call) (zero : α) (heap : CellHeap α) (j : Fin h.cols) :
    zeroWeight h zero heap ("dw",j.val)=zero := by simp [zeroWeight,j.isLt]

theorem zeroWeight_frame {α : Type} (h : Call) (zero : α) (heap : CellHeap α) (r : RegionName) (o : Nat)
    (outside : ¬(r="dw" ∧ o<h.cols)) : zeroWeight h zero heap (r,o)=heap (r,o) := by
  simp [zeroWeight,outside]

inductive Event | viewX | viewW | allocateY | allocateR | forwardGrid | saveContext
  | snapshotG | allocateDW | weightGrid | inputGrid | zeroDW
  deriving DecidableEq

def forwardEvents (h : Call) : List Event :=
  [.viewX,.viewW,.allocateY,.allocateR]++(if h.rows=0 then [] else [.forwardGrid])++[.saveContext]
def backwardEvents (h : Call) : List Event :=
  [.snapshotG,.allocateDW]++(if h.rows=0 then [.zeroDW] else [.weightGrid,.inputGrid])

theorem empty_no_launch (h : Call) (empty : h.rows=0) :
    forwardEvents h=[.viewX,.viewW,.allocateY,.allocateR,.saveContext] ∧
    backwardEvents h=[.snapshotG,.allocateDW,.zeroDW] := by simp [forwardEvents,backwardEvents,empty]

theorem backward_order (h : Call) (nonempty : h.rows≠0) :
    backwardEvents h=[.snapshotG,.allocateDW,.weightGrid,.inputGrid] := by simp [backwardEvents,nonempty]

#axiomsClean validated
#axiomsClean preparation_failure
#axiomsClean saved_addresses
#axiomsClean empty_no_launch
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.Host
