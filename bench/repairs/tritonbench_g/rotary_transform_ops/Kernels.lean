import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.RotaryTransformOps

structure Ops (α : Type) where
  zero : α
  one : α
  neg : α → α
  mul : α → α → α
  sub : α → α → α
  add : α → α → α

/-- Original source arithmetic, including the different high-output addition
order in the split and adjacent branches. Both halves are loaded first. -/
def sourcePair {α : Type} (p : Ops α) (interleaved conjugate : Bool)
    (a b c sine : α) : α × α :=
  let s := if conjugate then p.neg sine else sine
  (p.sub (p.mul a c) (p.mul b s),
    if interleaved then p.add (p.mul b c) (p.mul a s)
    else p.add (p.mul a s) (p.mul b c))
noncomputable def realOps : Ops ℝ := ⟨0,1,(-·),(·*·),(·-·),(·+·)⟩

inductive Region where
  | input | output | cosine | sine | cumulative | offsets | other (n : Nat)
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α
def component (v : α×α) (second : Bool) := if second then v.2 else v.1

structure Shape where
  batches : Nat
  maximum : Nat
  total : Nat
  heads : Nat
  width : Nat
  half : Nat
  coefficientRows : Nat
  varlen : Bool
  offsetTensor : Bool
  scalarOffset : Nat
  interleaved : Bool
  conjugate : Bool
  inplace : Bool
  deriving DecidableEq, Repr
structure Metadata where
  cumulative : Nat → Nat
  offsets : Nat → Nat
structure Config where
  shape : Shape
  metadata : Metadata

def Shape.output (s : Shape) : Region := if s.inplace then .input else .output
def Shape.blockK (s : Shape) := if 2*s.half≤32 then 32 else
  if 2*s.half≤64 then 64 else if 2*s.half≤128 then 128 else 256
def Shape.blockM (s : Shape) := if s.interleaved then 4 else if 2*s.half≤128 then 8 else 4
def Shape.rowBlocks (s : Shape) := (s.maximum+s.blockM-1)/s.blockM
def Shape.address (s : Shape) (token head col : Nat) := (token*s.heads+head)*s.width+col
def Config.start (c : Config) (b : Nat) := if c.shape.varlen then c.metadata.cumulative b else b*c.shape.maximum
def Config.length (c : Config) (b : Nat) := if c.shape.varlen then
  c.metadata.cumulative (b+1)-c.metadata.cumulative b else c.shape.maximum
def Config.shift (c : Config) (b : Nat) := if c.shape.offsetTensor then c.metadata.offsets b else c.shape.scalarOffset

structure Config.Valid (c : Config) : Prop where
  heads : 0<c.shape.heads
  width : 0<c.shape.width
  widthLimit : c.shape.width≤256
  rotary : 2*c.shape.half≤c.shape.width
  cumulativeZero : c.metadata.cumulative 0=0
  cumulativeMonotone : ∀ a b,a≤b → b≤c.shape.batches → c.metadata.cumulative a≤c.metadata.cumulative b
  total : c.shape.total=if c.shape.varlen then c.metadata.cumulative c.shape.batches else c.shape.batches*c.shape.maximum
  lengthBound : ∀ b,b<c.shape.batches → c.length b≤c.shape.maximum

structure Work where
  batch : Nat
  row : Nat
  head : Nat
  pair : Nat
  deriving DecidableEq, Repr
def Config.Active (c : Config) (w : Work) : Prop :=
  w.batch<c.shape.batches ∧ w.row<c.length w.batch ∧ w.head<c.shape.heads ∧ w.pair<c.shape.half
instance (c : Config) (w : Work) : Decidable (c.Active w) := inferInstanceAs (Decidable (_∧_∧_∧_))
def Shape.dim (s : Shape) (pair : Nat) (second : Bool) :=
  if s.interleaved then 2*pair+(if second then 1 else 0) else pair+(if second then s.half else 0)
def Config.cell (c : Config) (w : Work) (second : Bool) :=
  c.shape.address (c.start w.batch+w.row) w.head (c.shape.dim w.pair second)
def Config.coeff (c : Config) (w : Work) := (w.row+c.shift w.batch)*c.shape.half+w.pair
def Config.hasCoeff (c : Config) (w : Work) := w.row+c.shift w.batch<c.shape.coefficientRows
instance (c : Config) (w : Work) : Decidable (c.hasCoeff w) := inferInstanceAs (Decidable (_<_))

def pairValues {α : Type} (c : Config) (p : Ops α) (w : Work) (m : Memory α) :=
  sourcePair p c.shape.interleaved c.shape.conjugate
    (m .input (c.cell w false)) (m .input (c.cell w true))
    (if c.hasCoeff w then m .cosine (c.coeff w) else p.one)
    (if c.hasCoeff w then m .sine (c.coeff w) else p.zero)
def step {α : Type} (c : Config) (p : Ops α) (w : Work) (m : Memory α) : Memory α :=
  let v := pairValues c p w m
  let low := fun r a => if r=c.shape.output ∧ a=c.cell w false then v.1 else m r a
  fun r a => if r=c.shape.output ∧ a=c.cell w true then v.2 else low r a
def Writes (c : Config) (w : Work) (r : Region) (a : Nat) :=
  r=c.shape.output ∧ (a=c.cell w false ∨ a=c.cell w true)
instance (c : Config) (w : Work) (r : Region) (a : Nat) : Decidable (Writes c w r a) :=
  inferInstanceAs (Decidable (_∧(_∨_)))
end VeriTile.Bench.Repairs.RotaryTransformOps

