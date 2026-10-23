import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Physical

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
namespace HostABI

def tileRule (size tile : Nat) : Prop :=
  tile=if size ≤ 16 then 16 else if size ≤ 32 then 32 else 64

/-- Exact integer inventory checked by _plan before any launch. -/
def metadata (c : Shape) : List Nat :=
  [c.B,c.H,c.T,c.K,c.V,64,c.BK,c.BV,c.chunks,c.keyBlocks,c.valueBlocks,
    c.heads,c.paddedTime,c.paddedKey,c.paddedValue,c.T*c.K,c.T*c.V,
    c.chunks*c.K,c.chunks*c.K*c.V]

def keyExtent (c : Shape) : Nat := if 0<c.heads ∧ 0<c.chunks then
  (c.heads-1)*c.T*c.K+(c.paddedTime-1)*c.K+c.paddedKey else 0

def valueExtent (c : Shape) : Nat := if 0<c.heads ∧ 0<c.chunks then
  (c.heads-1)*c.T*c.V+(c.paddedTime-1)*c.V+c.paddedValue else 0

def historyExtent (c : Shape) : Nat := if 0<c.heads ∧ 0<c.chunks then
  (c.heads-1)*c.chunks*c.K*c.V+(c.chunks-1)*c.K*c.V+
    (c.paddedKey-1)*c.V+c.paddedValue else 0

def stateExtent (c : Shape) : Nat := if 0<c.heads then
  (c.heads-1)*c.K*c.V+(c.paddedKey-1)*c.V+c.paddedValue else 0

def partialExtent (c : Shape) : Nat := if 0<c.heads ∧ 0<c.chunks then
  (c.keyBlocks*c.heads-1)*c.T*c.V+(c.paddedTime-1)*c.V+c.paddedValue else 0

def extents (c : Shape) : List Nat :=
  [keyExtent c,valueExtent c,historyExtent c,stateExtent c,partialExtent c]

structure Accepted (c : Shape) : Prop where
  keyRule : tileRule c.K c.BK
  valueRule : tileRule c.V c.BV
  metadata32 : ∀ n∈metadata c,n<2^31
  extents64 : ∀ n∈extents c,4*n<2^63
  axes : 0<c.heads → c.heads ≤ 65535 ∧ c.chunks ≤ 65535 ∧ c.valueBlocks ≤ 65535

/-- The shared driver's signed grid-product behavior is explicitly outside this
program-mainline repair. This is ordinary legal launch completion, not a
numerical law and not a claim that the product check has been repaired. -/
def RuntimeContract (c : Shape) : Prop := c.heads=0 ∨
  (c.keyBlocks*c.valueBlocks*c.heads<2^31 ∧
    c.valueBlocks*c.chunks*c.heads<2^31 ∧ c.keyBlocks*c.chunks*c.heads<2^31)

structure Grid where
  x : Nat
  y : Nat
  z : Nat

def gridH (c : Shape) : Grid := ⟨c.keyBlocks,c.valueBlocks,c.heads⟩
def gridO (c : Shape) : Grid := ⟨c.valueBlocks,c.chunks,c.heads⟩
def gridQKV (c : Shape) : Grid := ⟨c.keyBlocks,c.chunks,c.heads⟩
def Grid.Legal (g : Grid) : Prop := g.x<2^31 ∧ g.y ≤ 65535 ∧ g.z ≤ 65535

theorem launch_axes (c : Shape) (ok : Accepted c) (active : 0<c.heads) :
    (gridH c).Legal ∧ (gridO c).Legal ∧ (gridQKV c).Legal := by
  have hk := ok.metadata32 c.keyBlocks (by simp [metadata])
  have hv := ok.metadata32 c.valueBlocks (by simp [metadata])
  have ha := ok.axes active
  exact ⟨⟨hk,ha.2.2,ha.1⟩,⟨hv,ha.2.1,ha.1⟩,⟨hk,ha.2.1,ha.1⟩⟩

theorem signed32 (n : Nat) (hn : n<2^31) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n:Int)) (by decide) (by omega) (by exact_mod_cast hn)

theorem signed64 (n : Int) (lo : -(2^63) ≤ n) (hi : n<2^63) :
    (BitVec.ofInt 64 n).toInt=n :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

theorem padded_key32 (c : Shape) (ok : Accepted c) (block lane : Nat)
    (hb : block<c.keyBlocks) (hl : lane<c.BK) :
    (BitVec.ofNat 32 (block*c.BK+lane)).toInt=((block*c.BK+lane:Nat):Int) := by
  apply signed32
  exact lt_trans (Shape.padded_lane_bound _ _ _ _ hb hl)
    (ok.metadata32 c.paddedKey (by simp [metadata]))

theorem padded_value32 (c : Shape) (ok : Accepted c) (block lane : Nat)
    (hb : block<c.valueBlocks) (hl : lane<c.BV) :
    (BitVec.ofNat 32 (block*c.BV+lane)).toInt=((block*c.BV+lane:Nat):Int) := by
  apply signed32
  exact lt_trans (Shape.padded_lane_bound _ _ _ _ hb hl)
    (ok.metadata32 c.paddedValue (by simp [metadata]))

theorem padded_time32 (c : Shape) (ok : Accepted c) (chunk lane : Nat)
    (hc : chunk<c.chunks) (hl : lane<64) :
    (BitVec.ofNat 32 (chunk*64+lane)).toInt=((chunk*64+lane:Nat):Int) := by
  apply signed32
  exact lt_trans (Shape.padded_lane_bound _ _ _ _ hc hl)
    (ok.metadata32 c.paddedTime (by simp [metadata]))

/-- Tensor views retain actual storage offsets. The original contiguous wrapper
normalizes logical values before validation; this contract does not replace it
or require the caller's original view to have canonical strides. -/
structure TensorMeta where
  dimensions : List Nat
  strides : List Nat
  storage : Nat
  offset : Nat
  capacity : Nat
  device : Nat
  fp32 : Bool
  cuda : Bool
  resolved : Bool
  contiguous : Bool

structure NormalizedView (α : Type) where
  before : TensorMeta
  after : TensorMeta
  beforeValue : List Nat → α
  afterValue : List Nat → α
  shape : after.dimensions=before.dimensions
  dtype : after.fp32=before.fp32
  device : after.device=before.device
  dense : after.contiguous=Bool.true
  coordinateCopy : ∀ i,afterValue i=beforeValue i

/-- Canonical allocation strides use max(dim,1), including empty tensors. -/
def strides4 (d0 d1 d2 d3 : Nat) : List Nat :=
  [max d1 1*max d2 1*max d3 1,max d2 1*max d3 1,max d3 1,1]

def shapeKey (c : Shape) := [c.B,c.H,c.T,c.K]
def shapeValue (c : Shape) := [c.B,c.H,c.T,c.V]
def shapeState (c : Shape) := [c.B,c.H,c.K,c.V]
def shapeHistory (c : Shape) := [c.B,c.H,c.chunks*c.K,c.V]

def effectiveInputs (xs : Inputs α) (zero : α) (seed incoming terminal : Bool) : Inputs α :=
  {xs with
    seed := (if seed then xs.seed else fun _ _ _ => zero)
    incoming := (if incoming then xs.incoming else fun _ _ _ => zero)
    finalIncoming := (if terminal then xs.finalIncoming else fun _ _ _ => zero)}

structure ForwardReturn (α : Type) where
  memory : Memory α
  output : TensorMeta
  finalState : Option TensorMeta
  savedQ : TensorMeta
  savedK : TensorMeta
  savedV : TensorMeta
  savedHistory : TensorMeta

structure BackwardReturn (α : Type) where
  memory : Memory α
  dq : TensorMeta
  dk : TensorMeta
  dv : TensorMeta
  initialGradient : Option TensorMeta
  flagGradient : Option TensorMeta

def detachedBackward (memory : Memory α) (dq dk dv : TensorMeta) : BackwardReturn α :=
  ⟨memory,dq,dk,dv,none,none⟩

theorem initial_detach (memory : Memory α) (dq dk dv : TensorMeta) :
    (detachedBackward memory dq dk dv).initialGradient=none ∧
    (detachedBackward memory dq dk dv).flagGradient=none := ⟨rfl,rfl⟩

#axiomsClean launch_axes
#axiomsClean signed32
#axiomsClean padded_key32
#axiomsClean padded_value32
#axiomsClean padded_time32
#axiomsClean initial_detach
end HostABI
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
