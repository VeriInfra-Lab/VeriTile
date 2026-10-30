import bench.tritonbench_g.max_reduction.Proofs.ReductionModel

namespace VeriTile.Bench.Repairs.MaxReduction.Native
noncomputable section
open Classical
set_option maxHeartbeats 1000000

def magnitude (x : Word) : Nat := x.toNat % 2^31
def sign (x : Word) : Bool := decide (2^31≤x.toNat)
def nan (x : Word) : Bool := decide (0x7f800000<magnitude x)
def zero (x : Word) : Bool := decide (magnitude x=0)
def equal (x y : Word) : Bool :=
  !nan x && !nan y && (decide (x=y) || (zero x && zero y))
def greater (x y : Word) : Bool :=
  !nan x && !nan y && !(zero x && zero y) &&
    (if sign x=sign y then
      if sign x then decide (magnitude x<magnitude y) else decide (magnitude y<magnitude x)
    else !sign x)

/-- The target's canonical NaN word is an explicit target-ABI parameter.
The maxnum obligation includes canonicalization/quieting and signed-zero ties;
these properties are not inferred from a mathematical real maximum. -/
structure Target where
  canonicalNaN : Word
  isNaN : nan canonicalNaN=Bool.true
  quiet : canonicalNaN.toNat / 2^22 % 2=1

def maximum (target : Target) (x y : Word) : Word :=
  if nan x then (if nan y then target.canonicalNaN else y)
  else if nan y then x
  else if greater x y then x
  else if greater y x then y
  else if zero x && zero y then x &&& y
  else y

/-- Exactly three local scalar FP32 obligations, for the selected native
profile and every pair of raw words (including infinities, zeros and NaNs).
There is no output, reduction, admission, index, memory, or runtime premise. -/
structure LocalObligations (actual : NativeOps) (target : Target) : Prop where
  maxnum32 : ∀ x y,actual.maximum x y=maximum target x y
  orderedEq32 : ∀ x y,actual.orderedEqual x y=equal x y
  orderedGt32 : ∀ x y,actual.orderedGreater x y=greater x y

def referencePair (left right : Word × Nat) : Word × Nat :=
  if greater left.1 right.1 || (equal left.1 right.1 && decide (left.2<right.2))
  then left else right

theorem pair_equiv (actual : NativeOps) (target : Target) (hlocal : LocalObligations actual target)
    (left right : Word × Nat) : nativePair actual left right=referencePair left right := by
  simp only [nativePair,referencePair,hlocal.orderedEq32,hlocal.orderedGt32]
  split <;> rfl

theorem tree_equiv (tree : Tree) (f g : A → A → A) (values other : Nat → A)
    (op : ∀ a b,f a b=g a b) (leaf : ∀ i,i∈tree.leaves → values i=other i) :
    tree.eval f values=tree.eval g other := by
  induction tree with
  | leaf i => exact leaf i (by simp [Tree.leaves])
  | node l r il ir =>
    simp only [Tree.eval]
    rw [il (fun i hi => leaf i (List.mem_append_left _ hi)),
      ir (fun i hi => leaf i (List.mem_append_right _ hi)),op]

def referenceStage (target : Target) (tree : Tree) (input : Nat → Word) (count start : Nat) : Word :=
  tree.eval (maximum target) (fun i => if start+i<count then input (start+i) else negativeInfinity)

def referenceDimension (tree : Tree) (input : Nat → Word) (N K m k : Nat) : Word × IndexWord :=
  let pair := tree.eval referencePair (fun n => (if n<N then input (m*N*K+n*K+k) else negativeInfinity,n))
  (pair.1,indexStore pair.2)

theorem stage_equiv (actual : NativeOps) (target : Target) (hlocal : LocalObligations actual target)
    (tree : Tree) (input : Nat → Word) (count start : Nat) :
    scalarReduction actual.maximum negativeInfinity tree input count start=
      referenceStage target tree input count start :=
  tree_equiv tree _ _ _ _ hlocal.maxnum32 (fun _ _ => rfl)

theorem dimension_equiv (actual : NativeOps) (target : Target) (hlocal : LocalObligations actual target)
    (tree : Tree) (input : Nat → Word) (N K m k : Nat) :
    nativeDimension actual tree input N K m k=referenceDimension tree input N K m k := by
  unfold nativeDimension referenceDimension
  rw [tree_equiv tree (nativePair actual) referencePair _ _ (pair_equiv actual target hlocal) (fun _ _ => rfl)]

/-- A raw NaN does not establish a valid attaining logical argmax. Ordered
comparisons select the right operand whenever both comparisons are false. -/
theorem unordered_select_right (actual : NativeOps) (left right : Word × Nat)
    (eqFalse : actual.orderedEqual left.1 right.1=Bool.false)
    (gtFalse : actual.orderedGreater left.1 right.1=Bool.false) :
    nativePair actual left right=right := by simp [nativePair,eqFalse,gtFalse]

#axiomsClean stage_equiv
#axiomsClean dimension_equiv
#axiomsClean unordered_select_right
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.Native
