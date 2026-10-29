import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.TritonArgmax
namespace Reduction

inductive Tree (n : Nat) where
  | leaf : Fin n → Tree n
  | fork : Tree n → Tree n → Tree n
  deriving Repr

def Tree.leaves {n : Nat} : Tree n → List (Fin n)
  | .leaf i => [i]
  | .fork a b => a.leaves ++ b.leaves

/-- The exact source comparator uses strict greater, equal, and lane tie-left.
No law on `gt` or `eq` is used by the structural execution. -/
def leftWins {α : Type} (gt eq : α → α → Bool)
    (x y : α) (i j : Nat) : Bool := gt x y || (eq x y && decide (i  <  j))

def Tree.pick {n : Nat} {α : Type} (gt eq : α → α → Bool)
    (xs : Fin n → α) : Tree n → Fin n
  | .leaf i => i
  | .fork a b =>
      let i := a.pick gt eq xs
      let j := b.pick gt eq xs
      if leftWins gt eq (xs i) (xs j) i.val j.val then i else j

def Tree.eval {n : Nat} {α : Type} (gt eq : α → α → Bool)
    (xs : Fin n → α) (t : Tree n) : α × Fin n :=
  let i := t.pick gt eq xs
  (xs i,i)

end Reduction
end VeriTile.Bench.Repairs.TritonArgmax

/-! The unchanged floating-operation source, with repaired i64 address/control
casts erased in the exact-integer projection. The full `dtype`-dependent raw
graph below keeps intermediate conversion and comparison as scalar operations;
it does not pretend integer/bool/fp64 storage is the original FP32 helper.
-/
namespace VeriTile.Bench.Repairs.TritonArgmax
open VeriTile.Triton
namespace Source

abbrev Word := BitVec 32

/-- The actual intermediate scalar format is retained, even when the shared
typed floating AST has no matching channel. `bytes` is its storage width.
Scalar callbacks are source primitives, never reduction or output oracles. -/
structure MidFormat where
  name : String
  bytes : Nat
  positive : 0 < bytes
  deriving Repr

abbrev MidWord (d : MidFormat) := BitVec (8*d.bytes)

structure Scalars (d : MidFormat) where
  gt32 : Word → Word → Bool
  eq32 : Word → Word → Bool
  negInf32 : Word
  storeCast : Word → MidWord d
  gtMid : MidWord d → MidWord d → Bool
  eqMid : MidWord d → MidWord d → Bool
  negInfMid : MidWord d

def blockM : Nat := 128
def blockN : Nat := 128

/-- `blocks` and `midBlock` are the actual host ceil/next-power-of-two values.
Their arithmetic properties are proved at host admission, rather than assumed
as a successful-kernel premise. The dynamic source block policy is retained. -/
structure GlobalConfig where
  size : Nat
  block : Nat
  blocks : Nat
  midBlock : Nat
  firstTree : Nat → Reduction.Tree block
  secondTree : Reduction.Tree midBlock

def GlobalConfig.loaded (c : GlobalConfig) {α : Type}
    (input : Nat → α) (padding : α) (pid : Nat) (lane : Fin c.block) : α :=
  if pid*c.block+lane.val < c.size then input (pid*c.block+lane.val) else padding

def GlobalConfig.selected (c : GlobalConfig) {α : Type}
    (gt eq : α → α → Bool) (input : Nat → α) (padding : α) (pid : Nat) : Fin c.block :=
  (c.firstTree pid).pick gt eq (c.loaded input padding pid)

def stage1Value {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (input : Nat → Word) (pid : Nat) : MidWord d :=
  ops.storeCast (c.loaded input ops.negInf32 pid
    (c.selected ops.gt32 ops.eq32 input ops.negInf32 pid))

def stage1Index {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (input : Nat → Word) (pid : Nat) : Nat :=
  (c.selected ops.gt32 ops.eq32 input ops.negInf32 pid).val+pid*c.block

def stage2Loaded {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (mid : Nat → MidWord d) (lane : Fin c.midBlock) : MidWord d :=
  if lane.val < c.blocks then mid lane.val else ops.negInfMid

def stage2Selected {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (mid : Nat → MidWord d) : Fin c.midBlock :=
  c.secondTree.pick ops.gtMid ops.eqMid (stage2Loaded ops c mid)

/-- `mid_index` has `midBlock` initialized entries, including selected padding.
Only stage1's actual `blocks` entries are overwritten by their source indices. -/
def globalIndex {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (input : Nat → Word) : Nat :=
  let selected := stage2Selected ops c (stage1Value ops c input)
  if selected.val < c.blocks then stage1Index ops c input selected.val else 0

structure DimConfig where
  outer : Nat
  extent : Nat
  inner : Nat
  tree : Nat → Nat → Nat → Reduction.Tree blockN

def dimInputOffset (c : DimConfig) (row column inner : Nat) : Nat :=
  row*c.extent*c.inner+column*c.inner+inner

def dimOutputOffset (c : DimConfig) (row inner : Nat) : Nat := row*c.inner+inner

def DimConfig.loaded (c : DimConfig) {α : Type} (input : Nat → α) (padding : α)
    (row inner tile : Nat) (lane : Fin blockN) : α :=
  let column := tile*blockN+lane.val
  if row < c.outer ∧ column < c.extent then input (dimInputOffset c row column inner) else padding

def dimLocal {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (input : Nat → Word) (row inner tile : Nat) : Word × Nat :=
  let values := c.loaded input ops.negInf32 row inner tile
  let selected := (c.tree row inner tile).pick ops.gt32 ops.eq32 values
  (values selected,tile*blockN+selected.val)

/-- FP32 running max and the strict update retain the first earlier tile on
ties. All tiles run; there is no single-block or non-NaN restriction. -/
def dimStep {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (input : Nat → Word) (row inner : Nat) (current : Word × Nat) (tile : Nat) : Word × Nat :=
  let selected := dimLocal ops c input row inner tile
  if ops.gt32 selected.1 current.1 then selected else current

def dimTiles (c : DimConfig) : List Nat := List.range ((c.extent+blockN-1)/blockN)

def dimIndex {d : MidFormat} (ops : Scalars d) (c : DimConfig)
    (input : Nat → Word) (row inner : Nat) : Nat :=
  ((dimTiles c).foldl (dimStep ops c input row inner) (ops.negInf32,0)).2

end Source
end VeriTile.Bench.Repairs.TritonArgmax
