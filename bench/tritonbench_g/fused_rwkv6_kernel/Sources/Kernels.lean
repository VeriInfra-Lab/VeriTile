import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 4000000

/-- Source sizes and its two independently passed input head pitches. -/
structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  keyPitch : Nat
  valuePitch : Nat
  initial : Bool
  final : Bool
  reverse : Bool
  deriving DecidableEq

def Shape.NK (c : Shape) := (c.K + c.BK - 1) / c.BK
def Shape.NV (c : Shape) := (c.V + c.BV - 1) / c.BV
def Shape.Z (c : Shape) := c.B * c.H

def visit (c : Shape) (n : Nat) := if c.reverse then c.T - 1 - n else n

/-- All numerical edges are scalar. A tree records every padded reduction leaf. -/
inductive Tree where
  | zero
  | leaf (i : Nat)
  | add (a b : Tree)
  deriving DecidableEq, Repr

def Tree.leaves : Tree → List Nat
  | .zero => []
  | .leaf i => [i]
  | .add a b => a.leaves ++ b.leaves

def Tree.eval {α : Type} (zero : α) (add : α → α → α) (f : Nat → α) : Tree → α
  | .zero => zero
  | .leaf i => f i
  | .add a b => add (a.eval zero add f) (b.eval zero add f)

def Tree.legal (t : Tree) (n : Nat) : Prop := t.leaves.Perm (List.range n)

def Tree.chain : Nat → Tree
  | 0 => .zero
  | n+1 => .add (chain n) (.leaf n)

@[simp] theorem Tree.chain_leaves (n : Nat) : (Tree.chain n).leaves = List.range n := by
  induction n with
  | zero => rfl
  | succ n ih => simpa [Tree.chain,Tree.leaves,ih] using (@List.range_succ n).symm

theorem Tree.chain_legal (n : Nat) : (Tree.chain n).legal n := by simp [Tree.legal]

structure Ops (α : Type) where
  zero : α
  loadCast : α → α
  storeCast : α → α
  add : α → α → α
  mul : α → α → α
  exp : α → α

/-- Runtime-selected legal trees may differ by source time, block and output lane.
Selection is structural and is fixed before values are supplied. -/
structure Plans (c : Shape) where
  key : Nat → Nat → Nat → Nat → Tree
  host : Nat → Nat → Nat → Tree
  keyLegal : ∀ z i n f, (key z i n f).legal c.BK
  hostLegal : ∀ z t f, (host z t f).legal c.NK

def sequentialPlans (c : Shape) : Plans c where
  key := fun _ _ _ _ => Tree.chain c.BK
  host := fun _ _ _ => Tree.chain c.NK
  keyLegal := fun _ _ _ _ => Tree.chain_legal _
  hostLegal := fun _ _ _ => Tree.chain_legal _

inductive Region where
  | q | k | v | w | u | h0 | o | ht | result | outside
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α

structure Program where
  z : Nat
  ik : Nat
  iv : Nat
  deriving DecidableEq, Repr

def keyIndex (c : Shape) (p : Program) (j : Nat) := p.ik*c.BK+j
def valueIndex (c : Shape) (p : Program) (l : Nat) := p.iv*c.BV+l

def inputKey (c : Shape) (z t e : Nat) := z*c.keyPitch+t*c.K+e
def inputValue (c : Shape) (z t f : Nat) := z*c.valuePitch+t*c.V+f
def biasAddress (c : Shape) (z e : Nat) := (z%c.H)*c.K+e
def stateAddress (c : Shape) (z e f : Nat) := (z*c.K+e)*c.V+f
/-- The repaired allocation pitch, independent of the input V pitch. -/
def partialAddress (c : Shape) (z i t f : Nat) := ((i*c.Z+z)*c.T+t)*c.V+f
def resultAddress (c : Shape) (z t f : Nat) := (z*c.T+t)*c.V+f

abbrev Carry (α : Type) := Nat → Nat → α

def load {α : Type} (a : Ops α) (m : Memory α) (r : Region) (address : Nat) (active : Bool) :=
  a.loadCast (if active then m r address else a.zero)

def initialCarry {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α) : Carry α :=
  fun l j => if c.initial then a.add a.zero (load a m .h0
    (stateAddress c p.z (keyIndex c p j) (valueIndex c p l))
    (decide (keyIndex c p j < c.K ∧ valueIndex c p l < c.V))) else a.zero

def query {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α) (scale : α) (n j : Nat) :=
  a.mul (load a m .q (inputKey c p.z (visit c n) (keyIndex c p j))
    (decide (keyIndex c p j < c.K))) scale

def outer {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α) (n l j : Nat) :=
  a.mul (load a m .k (inputKey c p.z (visit c n) (keyIndex c p j))
    (decide (keyIndex c p j < c.K)))
    (load a m .v (inputValue c p.z (visit c n) (valueIndex c p l))
    (decide (valueIndex c p l < c.V)))

def decay {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α) (n j : Nat) :=
  a.exp (load a m .w (inputKey c p.z (visit c n) (keyIndex c p j))
    (decide (keyIndex c p j < c.K)))

def bias {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α) (j : Nat) :=
  load a m .u (biasAddress c p.z (keyIndex c p j)) (decide (keyIndex c p j < c.K))

def nextCarry {α : Type} (a : Ops α) (c : Shape) (p : Program) (m : Memory α)
    (h : Carry α) (n : Nat) : Carry α := fun l j =>
  a.add (a.mul (h l j) (decay a c p m n j)) (outer a c p m n l j)

def partialValue {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (p : Program)
    (m : Memory α) (scale : α) (h : Carry α) (n l : Nat) :=
  (plan.key p.z p.ik n (valueIndex c p l)).eval a.zero a.add (fun j =>
    a.mul (a.add (h l j) (a.mul (outer a c p m n l j) (bias a c p m j)))
      (query a c p m scale n j))

/-- An actual store; all other addresses, including all input bytes, remain old. -/
def write {α : Type} (m : Memory α) (r : Region) (address : Nat) (v : α) : Memory α :=
  fun s b => if s=r ∧ b=address then v else m s b

/-- Store one complete masked row. Selection of the lane is unique on active lanes;
this is equivalent to a sequence of the source's pairwise-disjoint masked stores. -/
def storePartial {α : Type} (a : Ops α) (c : Shape) (p : Program) (n : Nat)
    (values : Nat → α) (m : Memory α) : Memory α :=
  fun r b => if r=.o ∧ ∃ l<c.BV, valueIndex c p l<c.V ∧
    b=partialAddress c p.z p.ik (visit c n) (valueIndex c p l) then
      a.storeCast (values (b - partialAddress c p.z p.ik (visit c n) (p.iv*c.BV)))
    else m r b

def storeFinal {α : Type} (a : Ops α) (c : Shape) (p : Program)
    (h : Carry α) (m : Memory α) : Memory α :=
  fun r b => if c.final ∧ r=.ht ∧ ∃ l<c.BV, ∃ j<c.BK,
      valueIndex c p l<c.V ∧ keyIndex c p j<c.K ∧
      b=stateAddress c p.z (keyIndex c p j) (valueIndex c p l) then
    a.storeCast (h (b%c.V-p.iv*c.BV) (b/c.V%c.K-p.ik*c.BK)) else m r b

structure Machine (α : Type) where
  memory : Memory α
  carry : Carry α

def step {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (s : Machine α) (n : Nat) : Machine α where
  memory := storePartial a c p n (partialValue a c plan p s.memory scale s.carry n) s.memory
  carry := nextCarry a c p s.memory s.carry n

/-- Explicit current-memory register loop, followed by the original optional store. -/
def runLoop {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) : Nat → Machine α
  | 0 => ⟨m,initialCarry a c p m⟩
  | n+1 => step a c plan p scale (runLoop a c plan p scale m n) n

def runProgram {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) : Memory α :=
  let s := runLoop a c plan p scale m c.T
  storeFinal a c p s.carry s.memory

def programs (c : Shape) : List Program :=
  (List.range c.Z).flatMap fun z => (List.range c.NK).flatMap fun i =>
    (List.range c.NV).map fun v => ⟨z,i,v⟩

def runGrid {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (scale : α)
    (ps : List Program) (m : Memory α) :=
  ps.foldl (fun current p => runProgram a c plan p scale current) m

def hostValue {α : Type} (a : Ops α) (c : Shape) (plan : Plans c)
    (m : Memory α) (z t f : Nat) :=
  (plan.host z t f).eval a.zero a.add (fun i => m .o (partialAddress c z i t f))

def hostSum {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (m : Memory α) : Memory α :=
  fun r b => if r=.result ∧ b<c.Z*c.T*c.V then
      a.storeCast (hostValue a c plan m (b/c.V/c.T) (b/c.V%c.T) (b%c.V)) else m r b

def forward {α : Type} (a : Ops α) (c : Shape) (plan : Plans c) (scale : α) (m : Memory α) :=
  hostSum a c plan (runGrid a c plan scale (programs c) m)

end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
