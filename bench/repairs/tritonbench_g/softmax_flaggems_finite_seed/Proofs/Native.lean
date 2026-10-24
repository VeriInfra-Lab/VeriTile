import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev Word := BitVec 32

/-- Numerical words occur only in leaves. Compiler choices inspect Shape. -/
inductive Shape where
  | leaf (region : RegionName) (address : Nat)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | negInf
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Shape)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Shape)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Shape)
  | fromNat (precision : Option ComputeDType) (n : Nat)
  | fromInt (precision : Option ComputeDType) (n : Int)
  | bits (word : Word)
  | load (a : Shape)
  deriving Inhabited
inductive Term where
  | leaf (region : RegionName) (address : Nat) (word : Word)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | negInf
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Term)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Term)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Term)
  | fromNat (precision : Option ComputeDType) (n : Nat)
  | fromInt (precision : Option ComputeDType) (n : Int)
  | bits (word : Word)
  | load (a : Term)
  deriving Inhabited

def shape : Term → Shape
  | .leaf r o _ => .leaf r o
  | .literal p d x => .literal p d x
  | .negInf => .negInf
  | .binary p d op a b => .binary p d op (shape a) (shape b)
  | .unary p op a => .unary p op (shape a)
  | .cast p src dst a => .cast p src dst (shape a)
  | .fromNat p n => .fromNat p n
  | .fromInt p n => .fromInt p n
  | .bits w => .bits w
  | .load a => .load (shape a)

inductive Tree (n : Nat) where
  | lane (i : Fin n)
  | neutral
  | join (left right : Tree n)
def Tree.leaves : Tree n → List (Option (Fin n))
  | .lane i => [some i]
  | .neutral => [none]
  | .join a b => a.leaves++b.leaves
structure Plan (n : Nat) where
  tree : Tree n
  padding : Nat
  valid : tree.leaves.Perm ((List.finRange n).map some++List.replicate padding none)

def Tree.expand (p : Option ComputeDType) (op : Binary) (xs : Fin n → Term) (neutral : Term) : Tree n → Term
  | .lane i => xs i
  | .neutral => neutral
  | .join a b => .binary p .real op (a.expand p op xs neutral) (b.expand p op xs neutral)

theorem Plan.lane_coverage (p : Plan n) (i : Fin n) : some i∈p.tree.leaves := by
  apply p.valid.mem_iff.mpr
  simp

def Tree.seeded (xs : List (Fin n)) : Tree n := xs.foldr (fun i t => .join (.lane i) t) .neutral

theorem Tree.seeded_leaves (xs : List (Fin n)) : (Tree.seeded xs).leaves=xs.map some++[none] := by
  induction xs with
  | nil => rfl
  | cons x xs ih => change some x::(Tree.seeded xs).leaves=some x::(xs.map some++[none]); rw [ih]

def Plan.seeded (n : Nat) : Plan n :=
  ⟨Tree.seeded (List.finRange n),1,by rw [Tree.seeded_leaves]; exact .refl _⟩

/-- Full layout, reduced lane topology and output coordinate select the actual
scalar reduction tree. Every lane occurs exactly once; padding is explicit.
There is no numerical reduction callback. -/
abbrev Reductions := (isMax : Bool) → (precision : Option ComputeDType) → (s : TileShape) →
  (ax : Fin s.length) → (keep : Bool) → (out : TileIndex (TileShape.reduceShape s ax keep)) →
  (Fin (TileShape.axisDim s ax) → Shape) → Plan (TileShape.axisDim s ax)

def inputs (s : TileShape) (ax : Fin s.length) (keep : Bool) (xs : Values Term .real s) :
    TileIndex (TileShape.reduceShape s ax keep) → Fin (TileShape.axisDim s ax) → Term :=
  match keep with
  | Bool.false => fun out k => xs (TileShape.insertAxisIndex s ax out k)
  | Bool.true => fun out k => xs (TileShape.replaceAxisIndex s ax out k)

def reduction (R : Reductions) (isMax : Bool) (p : Option ComputeDType) {s : TileShape}
    (ax : Fin s.length) (keep : Bool) (xs : Values Term .real s)
    (out : TileIndex (TileShape.reduceShape s ax keep)) : Term :=
  let leaves := inputs s ax keep xs out
  let plan := R isMax p s ax keep out (fun i => shape (leaves i))
  plan.tree.expand p (if isMax then .max else .add) leaves
    (if isMax then .negInf else .literal p .real 0)

def algebra (R : Reductions) : Algebra Term where
  literal := .literal
  negInf := .negInf
  binary := .binary
  unary := .unary
  cast := .cast
  fromNat := .fromNat
  fromInt := .fromInt
  fp32Bits := fun w => .bits w.bits
  fp32Load := .load
  reduceMax := fun p => reduction R Bool.true p
  reduceSum := fun p => reduction R Bool.false p

/-- Only individual numerical instructions/conversions are deferred. No
kernel, tensor, reduction or whole-result instruction exists here. -/
structure Instructions where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  fmaRN : Word → Word → Word → Word

inductive Fusion where
  | split | left | right | leftSwap | rightSwap
  | negativeLeftFirst | negativeLeftSecond | negativeRightFirst | negativeRightSecond
  deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion
  swapMultiply : Shape → Bool

/-- Only an actual adjacent FP32 mul/add or sub/mul pair may contract.
The schedule also records multiplication operand order, preserving NaN payload behavior. A schedule cannot
introduce a new expression, inspect raw words, or assert FMA equals split ops.
Explicit load/cast nodes remain boundaries. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => I.literal p d x
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .leftSwap,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | .rightSwap,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | _,_,_,a,b => I.binary p d .add (eval I S a) (eval I S b)
  | .binary p d .sub a b =>
    match S.fusion (.binary p d .sub (shape a) (shape b)),p,d,a,b with
    | .negativeLeftFirst,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x ^^^ 0x80000000) (eval I S y) (eval I S c)
    | .negativeLeftSecond,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S y) (eval I S x ^^^ 0x80000000) (eval I S c)
    | .negativeRightFirst,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S y ^^^ 0x80000000) (eval I S x) (eval I S c)
    | .negativeRightSecond,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x) (eval I S y ^^^ 0x80000000) (eval I S c)
    | _,_,_,a,b => I.binary p d .sub (eval I S a) (eval I S b)
  | .binary p d .mul a b =>
    if S.swapMultiply (.binary p d .mul (shape a) (shape b)) then
      I.binary p d .mul (eval I S b) (eval I S a)
    else I.binary p d .mul (eval I S a) (eval I S b)
  | .binary p d op a b => I.binary p d op (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast p src dst a => I.cast p src dst (eval I S a)
  | .fromNat p n => I.fromNat p n
  | .fromInt p n => I.fromInt p n
  | .bits w => w
  | .load a => eval I S a

theorem left_fused (I : Instructions) (S : Schedule) (a b c : Term)
    (h : S.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul (shape a) (shape b)) (shape c))=.left) :
    eval I S (.binary (some .fp32) .real .add (.binary (some .fp32) .real .mul a b) c)=
      I.fmaRN (eval I S a) (eval I S b) (eval I S c) := by simp [eval,shape,h]

theorem right_fused (I : Instructions) (S : Schedule) (a b c : Term)
    (h : S.fusion (.binary (some .fp32) .real .add (shape c)
      (.binary (some .fp32) .real .mul (shape a) (shape b)))=.right) :
    eval I S (.binary (some .fp32) .real .add c (.binary (some .fp32) .real .mul a b))=
      I.fmaRN (eval I S a) (eval I S b) (eval I S c) := by simp [eval,shape,h]

def observe (I : Instructions) (S : Schedule) (s : State Term) (r : RegionName) (o : Nat) :=
  eval I S ((s.mem r o).read .real)

#axiomsClean Plan.lane_coverage
#axiomsClean left_fused
#axiomsClean right_fused
#specNonCircular eval avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Native
