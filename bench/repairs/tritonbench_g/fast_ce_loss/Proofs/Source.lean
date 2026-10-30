import bench.repairs.tritonbench_g.fast_ce_loss.Kernels
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Grid

namespace VeriTile.Bench.Repairs.FastCeLoss.Source
open VeriTile.Bench.Repairs.CrossEntropy1
noncomputable section

/-- Scalar operations and source reduction sites. FP interpretation expands every
reduction into its selected scalar graph; no reduction equality is a premise. -/
structure Arithmetic (α : Type) where
  literal : ℝ → α
  natural : Nat → α
  negInf : α
  load : α → α
  cast : α → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  exp : α → α
  log : α → α
  tanh : α → α
  maximum : {n : Nat} → (Fin n → α) → α
  sum : {n : Nat} → (Fin n → α) → α
  hostSum : {n : Nat} → (Fin n → α) → α
  hostLSE : {n : Nat} → (Fin n → α) → α
  hostAdd : α → α → α
  hostDiv : α → α → α
  upstreamDiv : α → α → α

structure Options where
  softcap : ℝ
  scaling : ℝ
  cap : Bool
  scale : Bool
  capSource : cap=decide (softcap≠0)
  scaleSource : scale=decide (scaling≠0)

structure Shape where
  batch : Nat
  sequence : Nat
  vocab : Nat
  block : Nat

def Shape.rows (s : Shape) : Nat := s.batch*s.sequence

def ceilDiv (n block : Nat) : Nat := n/block + if n%block=0 then 0 else 1

def Shape.chunks (s : Shape) : Nat := ceilDiv s.vocab 65536

def Shape.blocks (s : Shape) : Nat := ceilDiv s.vocab 4096

/-- Storage identity and contiguous base offset; object identity is retained
separately so returning a fresh allocation cannot satisfy the in-place return. -/
structure Tensor where
  object : Nat
  storage : Nat
  offset : Nat
  shape : List Nat
  deriving DecidableEq

structure Buffers where
  logits : Tensor
  labels : Tensor
  losses : Tensor
  chunks : Tensor
  lse : Tensor
  scalar : Tensor
  upstream : Tensor
  upstreamStride : Nat

abbrev Address := Nat × Nat
abbrev Memory (α : Type) := Address → α
abbrev Task := Nat × Nat

def addr (t : Tensor) (index : Nat) : Address := (t.storage,t.offset+index)

def tasks (rows blocks : Nat) : List Task := CrossEntropy1.Grid.tasks rows blocks

def transform {α : Type} (a : Arithmetic α) (o : Options) (x : α) : α :=
  let scaled := if o.scale then a.mul (a.literal o.scaling) x else x
  if o.cap then a.mul (a.literal o.softcap) (a.tanh (a.div scaled (a.literal o.softcap))) else scaled

def tile {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (m : Memory α) (q : Task) : Fin s.block → α := fun lane =>
  let col := q.2*s.block+lane.val
  let x := if col<s.vocab then a.load (m (addr b.logits (q.1*s.vocab+col))) else a.negInf
  if col<s.vocab then a.cast (transform a o x) else a.negInf

def stableLSE {α : Type} (a : Arithmetic α) {n : Nat} (values : Fin n → α) : α :=
  let maximum := a.maximum values
  a.add maximum (a.log (a.sum (fun i => a.exp (a.sub (values i) maximum))))

def forwardLSE {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (m : Memory α) (q : Task) : α := stableLSE a (tile a o s b m q)

def target {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (row : Nat) : α :=
  let x := a.load (m (addr b.logits (row*s.vocab+(labels row).toNat)))
  a.cast (transform a o (if s.chunks=1 then x else a.cast x))

def forwardLoss {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (q : Task) : α :=
  if labels q.1 = -100 then a.literal 0
  else if s.chunks=1 then a.sub (forwardLSE a o s b m q) (target a o s b labels m q.1)
  else a.mul (a.literal (-1)) (target a o s b labels m q.1)

def forwardWrites (s : Shape) (b : Buffers) (q : Task) (point : Address) : Prop :=
  point=addr b.chunks (q.1*s.chunks+q.2) ∨ (q.2=0 ∧ point=addr b.losses q.1)

instance (s : Shape) (b : Buffers) (q : Task) : DecidablePred (forwardWrites s b q) :=
  by
  intro point
  unfold forwardWrites
  infer_instance

def forwardValues {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (q : Task) (point : Address) : α :=
  if point=addr b.losses q.1 then forwardLoss a o s b labels m q else forwardLSE a o s b m q

def forwardStep {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (q : Task) (m : Memory α) : Memory α :=
  Schedule.put (forwardWrites s b) (forwardValues a o s b labels m) q m

def forwardGrid {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) : Memory α :=
  Schedule.run (forwardStep a o s b labels) (tasks s.rows s.chunks) m

def globalLSE {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (m : Memory α) (row : Nat) : α :=
  if s.chunks=1 then a.load (m (addr b.chunks row))
  else a.hostLSE (fun i : Fin s.chunks => a.load (m (addr b.chunks (row*s.chunks+i.val))))

def finishWrites (b : Buffers) (row : Nat) (point : Address) : Prop :=
  point=addr b.lse row ∨ point=addr b.losses row
instance (b : Buffers) (row : Nat) : DecidablePred (finishWrites b row) := by
  intro point
  unfold finishWrites
  infer_instance

def finishValues {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (row : Nat) (point : Address) : α :=
  if point=addr b.lse row then globalLSE a s b m row
  else if s.chunks=1 then m (addr b.losses row)
  else if labels row = -100 then a.literal 0
  else a.hostAdd (a.load (m (addr b.losses row))) (a.load (globalLSE a s b m row))

def finishStep {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (row : Nat) (m : Memory α) : Memory α :=
  Schedule.put (finishWrites b) (finishValues a s b labels m) row m

def finishGrid {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) : Memory α :=
  if s.chunks=1 then m else Schedule.run (finishStep a s b labels) (List.range s.rows) m

def count (s : Shape) (labels : Nat → Int) : Nat :=
  ((Finset.range s.rows).filter (fun r => labels r ≠ -100)).card

def denominator {α : Type} (a : Arithmetic α) (s : Shape) (labels : Nat → Int)
    (items : Option α) : α := items.getD (a.natural (count s labels))

def scalarLoss {α : Type} (a : Arithmetic α) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (items : Option α) (m : Memory α) : α :=
  a.hostDiv (a.load (a.hostSum (fun r : Fin s.rows => a.load (m (addr b.losses r.val)))))
    (denominator a s labels items)

/-- The host sum/division chain supplies one scalar upstream to every row;
this is the ordinary sum and scalar division autograd rule. -/
def rowUpstream {α : Type} (a : Arithmetic α) (s : Shape) (labels : Nat → Int)
    (items : Option α) (upstream : α) : α := a.upstreamDiv (a.load upstream) (denominator a s labels items)

def backwardScalar {α : Type} (a : Arithmetic α) (o : Options)
    (label : Int) (col : Nat) (x lse dloss : α) : α :=
  let x := a.load x
  let x := if o.scale then a.mul x (a.literal o.scaling) else x
  let partialValue := a.tanh (a.div x (a.literal o.softcap))
  let x := if o.cap then a.mul (a.literal o.softcap) partialValue else x
  let y := a.exp (a.sub (a.cast x) (a.load lse))
  let y := if (col : Int)=label then a.sub y (a.literal 1) else y
  let y := if o.scale then a.mul y (a.literal o.scaling) else y
  let y := if o.cap then a.mul y (a.sub (a.literal 1) (a.mul partialValue partialValue)) else y
  a.cast (if label = -100 then a.literal 0 else
    a.mul (if label = -100 then a.literal 0 else a.load dloss) y)

/-- The explicit source selection returns zero independently of every input
word and of any nonfinite intermediate. No multiplication-by-zero law is used. -/
theorem backward_ignored {α : Type} (a : Arithmetic α) (o : Options)
    (col : Nat) (x lse dloss : α) :
    backwardScalar a o (-100) col x lse dloss=a.cast (a.literal 0) := by
  simp [backwardScalar]

def backwardWrites (s : Shape) (b : Buffers) (q : Task) (point : Address) : Prop :=
  point.1=b.logits.storage ∧ b.logits.offset+q.1*s.vocab+q.2*4096≤point.2 ∧
  point.2<b.logits.offset+q.1*s.vocab+min ((q.2+1)*4096) s.vocab
instance (s : Shape) (b : Buffers) (q : Task) : DecidablePred (backwardWrites s b q) :=
  by
  intro point
  unfold backwardWrites
  infer_instance

def backwardValues {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) (q : Task) (point : Address) : α :=
  let col := point.2-(b.logits.offset+q.1*s.vocab)
  backwardScalar a o (labels q.1) col (m point) (m (addr b.lse q.1)) (m (addr b.upstream (q.1*b.upstreamStride)))

def backwardStep {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (q : Task) (m : Memory α) : Memory α :=
  Schedule.put (backwardWrites s b) (backwardValues a o s b labels m) q m

def backwardGrid {α : Type} (a : Arithmetic α) (o : Options) (s : Shape) (b : Buffers)
    (labels : Nat → Int) (m : Memory α) : Memory α :=
  Schedule.run (backwardStep a o s b labels) (tasks s.rows s.blocks) m

def backwardTuple (b : Buffers) : List (Option Tensor) := [some b.logits,none,none,none]

end
end VeriTile.Bench.Repairs.FastCeLoss.Source
