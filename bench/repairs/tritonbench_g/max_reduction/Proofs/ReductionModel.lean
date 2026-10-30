import bench.repairs.tritonbench_g.max_reduction.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaxReduction
open VeriTile.Triton
noncomputable section
open Classical

/-- A backend-selected reduction tree. Leaves include masked padding. No
associativity, sequential fold, or NaN permutation is imposed on native runs. -/
inductive Tree where
  | leaf (lane : Nat)
  | node (left right : Tree)
  deriving DecidableEq, Repr

def Tree.leaves : Tree → List Nat
  | .leaf i => [i]
  | .node l r => l.leaves++r.leaves

def Tree.eval (combine : A → A → A) (values : Nat → A) : Tree → A
  | .leaf i => values i
  | .node l r => combine (l.eval combine values) (r.eval combine values)

structure ReductionPlan (width : Nat) where
  tree : Tree
  complete : ∀ i,i∈tree.leaves ↔ i<width

/-- Runtime binder count types are int64, while the original indexed reducer
creates its local index in int32 and sign-extends when storing int64. -/
abbrev Word := BitVec 32
abbrev IndexWord := BitVec 64
structure NativeOps where
  maximum : Word → Word → Word
  orderedEqual : Word → Word → Bool
  orderedGreater : Word → Word → Bool

def negativeInfinity : Word := BitVec.ofNat 32 0xff800000

def nativePair (ops : NativeOps) (left right : Word × Nat) : Word × Nat :=
  let tie := ops.orderedEqual left.1 right.1 && decide (left.2<right.2)
  let gt := ops.orderedGreater left.1 right.1 || tie
  (if gt then left.1 else right.1,if gt then left.2 else right.2)

def indexStore (index : Nat) : IndexWord :=
  (BitVec.ofNat 32 index).signExtend 64

def scalarReduction (combine : A → A → A) (padding : A) (tree : Tree)
    (input : Nat → A) (count start : Nat) : A :=
  tree.eval combine (fun lane => if start+lane<count then input (start+lane) else padding)

def rowReduction (combine : A → A → A) (padding : A) (tree : Tree)
    (input : Nat → A) (N K m k : Nat) : A :=
  tree.eval combine (fun n => if n<N then input (m*N*K+n*K+k) else padding)

def nativeDimension (ops : NativeOps) (tree : Tree) (input : Nat → Word)
    (N K m k : Nat) : Word × IndexWord :=
  let pair := tree.eval (nativePair ops)
    (fun n => (if n<N then input (m*N*K+n*K+k) else negativeInfinity,n))
  (pair.1,indexStore pair.2)


end
end VeriTile.Bench.Repairs.MaxReduction
