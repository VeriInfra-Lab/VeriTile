import Mathlib
import VeriTile.Triton.Source
import VeriTile.Triton.Float.ScalarOps

/-! Pure launch data and independent ordered mathematical reference models.
Forward/reverse carries, indexed grid stores and host reductions are defined
from input arrays. Actual Triton quotations live only in Kernels.lean. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

structure Arithmetic (α : Type) where
  zero : α
  one : α
  negFive : α
  fromNat : Nat → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  exp2 : α → α
  scale : Nat → α

def realArithmetic : Arithmetic ℝ where
  zero := 0
  one := 1
  negFive := -5
  fromNat := Nat.cast
  add := (·+·)
  sub := (·-·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  exp2 := fun x => Real.exp (x*Real.log 2)
  scale := fun n => (Real.sqrt (n : ℝ))⁻¹

def decay (a : Arithmetic α) (head : Nat) : α :=
  a.sub a.one (a.exp2 (a.sub a.negFive (a.mul (a.fromNat head) a.one)))

def foldN (step : Nat → β → β) : Nat → β → β
  | 0,z => z
  | n+1,z => step n (foldN step n z)

inductive DotPlan where
  | product (index : Nat)
  | add (left right : DotPlan)
  | fma (index : Nat) (tail : DotPlan)
  deriving Repr

def DotPlan.leaves : DotPlan → List Nat
  | .product i => [i]
  | .add l r => l.leaves++r.leaves
  | .fma i t => i::t.leaves

def DotPlan.eval (a : Arithmetic α) (x y : Nat → α) : DotPlan → α
  | .product i => a.mul (x i) (y i)
  | .add l r => a.add (l.eval a x y) (r.eval a x y)
  | .fma i t => a.fma (x i) (y i) (t.eval a x y)

def DotPlan.Covers (p : DotPlan) (n : Nat) : Prop := p.leaves.Perm (List.range n)

inductive SumPlan where
  | zero
  | input (index : Nat)
  | add (left right : SumPlan)
  deriving Repr

def SumPlan.leaves : SumPlan → List Nat
  | .zero => []
  | .input i => [i]
  | .add l r => l.leaves++r.leaves

def SumPlan.eval (a : Arithmetic α) (x : Nat → α) : SumPlan → α
  | .zero => a.zero
  | .input i => x i
  | .add l r => a.add (l.eval a x) (r.eval a x)

def SumPlan.Covers (p : SumPlan) (n : Nat) : Prop := p.leaves.Perm (List.range n)

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  DK : Nat
  DV : Nat
  deriving DecidableEq

def tileCount (n width : Nat) : Nat := (n-1)/width+1

def Shape.BK (s : Shape) := min s.DK 32

def Shape.BV (s : Shape) := min s.DV 32

def Shape.NK (s : Shape) := tileCount s.DK s.BK

def Shape.NV (s : Shape) := tileCount s.DV s.BV

def Shape.BH (s : Shape) := s.B*s.H

def Shape.Admitted (s : Shape) : Prop :=
  0<s.DK ∧ 0<s.DV ∧ s.BK∈({1,2,4,8,16,32} : Finset Nat) ∧
    s.BV∈({1,2,4,8,16,32} : Finset Nat)

def rowAddress (s : Shape) (width head time channel : Nat) : Nat :=
  (head*s.T+time)*width+channel

def stateOffset (s : Shape) (head key value : Nat) : Nat :=
  (head*s.DK+key)*s.DV+value

def partialOffset (s : Shape) (width split head time channel : Nat) : Nat :=
  ((split*s.BH+head)*s.T+time)*width+channel

structure Inputs (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α
  do_ : Nat → Nat → Nat → α
  seed : Nat → Nat → Nat → α
  finalGrad : Nat → Nat → Nat → α

def keyLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h t k : Nat) : α :=
  if k<s.DK then f h t k else a.zero

def valueLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h t v : Nat) : α :=
  if v<s.DV then f h t v else a.zero

def stateLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h k v : Nat) : α :=
  if k<s.DK ∧ v<s.DV then f h k v else a.zero

structure Plans where
  output : DotPlan
  dq : DotPlan
  dk : DotPlan
  dv : DotPlan
  sumO : Nat → SumPlan
  sumQ : Nat → SumPlan
  sumK : Nat → SumPlan
  sumV : Nat → SumPlan

structure Plans.Valid (p : Plans) (s : Shape) : Prop where
  output : p.output.Covers s.BK
  dq : p.dq.Covers s.BV
  dk : p.dk.Covers s.BV
  dv : p.dv.Covers s.BK
  sumO : ∀ i,(p.sumO i).Covers s.NK
  sumQ : ∀ i,(p.sumQ i).Covers s.NV
  sumK : ∀ i,(p.sumK i).Covers s.NV
  sumV : ∀ i,(p.sumV i).Covers s.NK

abbrev Address := Nat × Int

abbrev Memory (α : Type) := Address → α

structure Tensor4 where
  storage : Nat
  offset : Nat
  capacity : Nat
  device : Nat
  strides : Fin 4 → Nat
  deriving DecidableEq

def Tensor4.address (a : Tensor4) (b h t d : Nat) : Address :=
  (a.storage,(a.offset+b*a.strides 0+h*a.strides 1+t*a.strides 2+d*a.strides 3 : Nat))

structure InputBuffers where
  q : Tensor4
  k : Tensor4
  v : Tensor4
  do_ : Tensor4
  seed : Tensor4
  finalGrad : Tensor4

def InputBuffers.qAddress (b : InputBuffers) (s : Shape) (h t k : Nat) : Address :=
  (b.q.storage,(b.q.offset+h*b.q.strides 1+t*s.DK+k : Nat))

def InputBuffers.kAddress (b : InputBuffers) (s : Shape) (h t k : Nat) : Address :=
  (b.k.storage,(b.k.offset+h*b.q.strides 1+t*s.DK+k : Nat))

def InputBuffers.vAddress (b : InputBuffers) (s : Shape) (h t v : Nat) : Address :=
  (b.v.storage,(b.v.offset+h*b.v.strides 1+t*s.DV+v : Nat))

def InputBuffers.doAddress (b : InputBuffers) (s : Shape) (h t v : Nat) : Address :=
  b.do_.address (h/s.H) (h%s.H) t v

def InputBuffers.seedAddress (b : InputBuffers) (s : Shape) (h k v : Nat) : Address :=
  (b.seed.storage,(b.seed.offset+stateOffset s h k v : Nat))

def InputBuffers.finalAddress (b : InputBuffers) (s : Shape) (h k v : Nat) : Address :=
  b.finalGrad.address (h/s.H) (h%s.H) k v

def InputBuffers.observe (b : InputBuffers) (s : Shape) (m : Memory α) : Inputs α where
  q := fun h t k => m (b.qAddress s h t k)
  k := fun h t k => m (b.kAddress s h t k)
  v := fun h t v => m (b.vAddress s h t v)
  do_ := fun h t v => m (b.doAddress s h t v)
  seed := fun h k v => m (b.seedAddress s h k v)
  finalGrad := fun h k v => m (b.finalAddress s h k v)

def InputBuffers.storageIds (b : InputBuffers) : Finset Nat :=
  {b.q.storage,b.k.storage,b.v.storage,b.do_.storage,b.seed.storage,b.finalGrad.storage}

def writeWord (m : Memory α) (p : Address) (x : α) : Memory α :=
  fun q => if q=p then x else m q

def fill (storage : Nat) (value : Nat → α) (items : List Nat) (m : Memory α) : Memory α :=
  items.foldl (fun current (i : Nat) => writeWord current (storage,(i : Int)) (value i)) m

def fillRange (storage count : Nat) (value : Nat → α) (m : Memory α) : Memory α :=
  fill storage value (List.range count) m

structure Allocation (n : Nat) (caller : Set Nat) where
  storage : Fin n → Nat
  distinct : Function.Injective storage
  fresh : ∀ i,storage i∉caller

def flatChannel (width i : Nat) := i%width

def flatTime (s : Shape) (width i : Nat) := i/width%s.T

def flatHead (s : Shape) (width i : Nat) := i/width/s.T%s.BH

def flatSplit (s : Shape) (width i : Nat) := i/width/s.T/s.BH

def rowCount (s : Shape) (width : Nat) := s.BH*s.T*width

inductive PartialKind where
  | output | dq | dk | dv | finalState
  deriving DecidableEq

def PartialKind.width (kind : PartialKind) (s : Shape) : Nat :=
  if kind=.dq ∨ kind=.dk then s.DK else s.DV

structure ProgramId where
  iv : Nat
  ik : Nat
  head : Nat
  deriving DecidableEq

def ProgramId.Valid (p : ProgramId) (s : Shape) := p.iv<s.NV ∧ p.ik<s.NK ∧ p.head<s.BH

def ProgramId.key (p : ProgramId) (s : Shape) (lane : Nat) := p.ik*s.BK+lane

def ProgramId.value (p : ProgramId) (s : Shape) (lane : Nat) := p.iv*s.BV+lane

abbrev TileState (α : Type) := Nat → Nat → α

def programForwardCarry (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (n : Nat) : TileState α :=
  foldN (fun t old lk lv => a.fma (decay a (p.head%s.H)) (old lk lv)
    (a.mul (keyLoad a s x.k p.head t (p.key s lk)) (valueLoad a s x.v p.head t (p.value s lv)))) n
    (fun lk lv => if seed then a.add a.zero (stateLoad a s x.seed p.head (p.key s lk) (p.value s lv)) else a.zero)

def programReverseCarry (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (n : Nat) : TileState α :=
  foldN (fun r old lk lv =>
    let t := s.T-1-r
    a.mul (a.fma (a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))
      (valueLoad a s x.do_ p.head t (p.value s lv)) (old lk lv)) (decay a (p.head%s.H))) n
    (fun lk lv => if final then stateLoad a s x.finalGrad p.head (p.key s lk) (p.value s lv) else a.zero)

def programAdjoint (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk lv : Nat) : α :=
  a.fma (a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))
    (valueLoad a s x.do_ p.head t (p.value s lv))
    (programReverseCarry a s x final p (s.T-1-t) lk lv)

def programO (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lv : Nat) : α :=
  plans.output.eval a (fun lk => programForwardCarry a s x seed p (t+1) lk lv)
    (fun lk => a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))

def programQ (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lk : Nat) : α :=
  a.mul (plans.dq.eval a (fun lv => programForwardCarry a s x seed p (t+1) lk lv)
    (fun lv => valueLoad a s x.do_ p.head t (p.value s lv))) (a.scale s.DK)

def programK (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk : Nat) : α :=
  plans.dk.eval a (fun lv => programAdjoint a s x final p t lk lv)
    (fun lv => valueLoad a s x.v p.head t (p.value s lv))

def programV (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lv : Nat) : α :=
  plans.dv.eval a (fun lk => programAdjoint a s x final p t lk lv)
    (fun lk => keyLoad a s x.k p.head t (p.key s lk))

def reduction (a : Arithmetic α) (plan : Nat → SumPlan) (input output count : Nat) (m : Memory α) : Memory α :=
  fillRange output count (fun offset => (plan offset).eval a (fun split => m (input,((split*count+offset : Nat) : Int)))) m

def sourceCell (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (kind : PartialKind) (i : Nat) : α :=
  let width := kind.width s
  let h := flatHead s width i
  let t := flatTime s width i
  let channel := flatChannel width i
  let split := flatSplit s width i
  match kind with
  | .output => programO a s p x seed ⟨channel/s.BV,split,h⟩ t (channel%s.BV)
  | .dq => programQ a s p x seed ⟨split,channel/s.BK,h⟩ t (channel%s.BK)
  | .dk => programK a s p x final ⟨split,channel/s.BK,h⟩ t (channel%s.BK)
  | .dv => programV a s p x final ⟨channel/s.BV,split,h⟩ t (channel%s.BV)
  | .finalState =>
      let h := i/s.DV/s.DK
      let k := i/s.DV%s.DK
      let v := i%s.DV
      programForwardCarry a s x seed ⟨v/s.BV,k/s.BK,h⟩ s.T (k%s.BK) (v%s.BV)

def sourcePhase (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α) : Memory α :=
  fillRange storage count (sourceCell a s p (b.observe s m) seed final kind) m

def sourceForward (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let intermediate := sourcePhase a s p b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV) m
  let intermediate := if final then sourcePhase a s p b seed Bool.false .finalState (alloc.storage 1)
      (s.BH*s.DK*s.DV) intermediate else intermediate
  reduction a p.sumO (alloc.storage 0) (alloc.storage 2) (rowCount s s.DV) intermediate

def sourceBackward (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let dq := sourcePhase a s p b seed final .dq (alloc.storage 0) (s.NV*rowCount s s.DK) m
  let dk := sourcePhase a s p b seed final .dk (alloc.storage 1) (s.NV*rowCount s s.DK) dq
  let dv := sourcePhase a s p b seed final .dv (alloc.storage 2) (s.NK*rowCount s s.DV) dk
  let dq := reduction a p.sumQ (alloc.storage 0) (alloc.storage 3) (rowCount s s.DK) dv
  let dk := reduction a p.sumK (alloc.storage 1) (alloc.storage 4) (rowCount s s.DK) dq
  reduction a p.sumV (alloc.storage 2) (alloc.storage 5) (rowCount s s.DV) dk

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
