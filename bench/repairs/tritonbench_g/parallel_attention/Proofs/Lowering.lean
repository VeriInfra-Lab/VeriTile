import bench.repairs.tritonbench_g.parallel_attention.Proofs.Graph

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Resolved scalar instruction shapes. These are finite syntax, not numerical
callbacks, and retain compiler contraction and operand order. -/
inductive ScaleForm where | right | left deriving Repr,DecidableEq
inductive TwiceForm where | multiplyLeft | multiplyRight | addSelf deriving Repr,DecidableEq
inductive ScaleTwiceForm where | separate | addSelf | fmaLeft | fmaRight deriving Repr,DecidableEq
inductive ScaledSumForm where | separate | fmaLeft | fmaRight | fmaOnZLeft | fmaOnZRight deriving Repr,DecidableEq

/-- The optimizer may reverse the final coefficient operands and move the
causal/validity select across doubling or the final product. Those instruction
positions are explicit syntax; no floating-point zero law is assumed. -/
inductive ProductOrder where | direct | swapped deriving Repr,DecidableEq
inductive MaskPosition where | beforeTwice | afterTwice | afterProduct deriving Repr,DecidableEq

structure ScalarLayout where
  scale : List Nat → ScaleForm
  twice : List Nat → TwiceForm
  scaleTwice : List Nat → ScaleTwiceForm
  scaledSum : List Nat → ScaledSumForm
  coefficientOrder : List Nat → ProductOrder
  coefficientMask : List Nat → MaskPosition

/-- A selected compiler layout includes the exact finite dot/reduction data
and scalar instruction choices. Coverage is a Real-algebra property, not a
claim that an arbitrary choice matches a compiler artifact. -/
structure CompilerLayout where
  plans : Plans
  scalar : ScalarLayout
  normTree : List Nat → Nat → DotPlan
  normComplete : ∀ site n,(normTree site n).leaves.Perm (List.range n)
  normNoCarry : ∀ site n,(normTree site n).carries=0

/-- A normalizer tree keeps source-rounded square leaves and fused square
accumulations distinct. Causal selects are retained on square leaves; fused
leaves use the selected raw value. The selected layout must bind these choices
to compiler syntax; coverage alone makes no such claim. -/
def normEval (a : Ops α) (raw : Nat → α) (masked : Nat → Bool) : DotPlan → α
  | .zero | .acc => a.zero
  | .product i => if masked i then a.zero else a.mul (raw i) (raw i)
  | .add l r => a.add (normEval a raw masked l) (normEval a raw masked r)
  | .fma i t => let x := if masked i then a.zero else raw i
               a.fma x x (normEval a raw masked t)

def nativeNorm (a : Ops α) (c : CompilerLayout) (site : List Nat) (n : Nat)
    (raw : Nat → α) (masked : Nat → Bool) : α := normEval a raw masked (c.normTree site n)

def liveDot (a : Ops α) (p : Plans) (site : List Nat) (n : Nat)
    (x y : Nat → α) (carry : α) : α := (p.dotTree site n).eval a x y carry

def ScaleForm.eval (f : ScaleForm) (a : Ops α) (x scale : α) : α :=
  match f with | .right => a.mul x scale | .left => a.mul scale x

def TwiceForm.eval (f : TwiceForm) (a : Ops α) (x : α) : α :=
  match f with
  | .multiplyLeft => a.mul a.two x
  | .multiplyRight => a.mul x a.two
  | .addSelf => a.add x x

def ScaleTwiceForm.eval (f : ScaleTwiceForm) (a : Ops α) (x scale : α) : α :=
  match f with
  | .separate => a.mul a.two (a.mul x scale)
  | .addSelf => a.add (a.mul scale x) (a.mul scale x)
  | .fmaLeft => a.fma scale x (a.mul scale x)
  | .fmaRight => a.fma x scale (a.mul x scale)

def ScaledSumForm.eval (f : ScaledSumForm) (a : Ops α) (x z scale : α) : α :=
  match f with
  | .separate => a.add (a.mul x scale) (a.mul z scale)
  | .fmaLeft => a.fma scale x (a.mul scale z)
  | .fmaRight => a.fma x scale (a.mul z scale)
  | .fmaOnZLeft => a.fma scale z (a.mul scale x)
  | .fmaOnZRight => a.fma z scale (a.mul x scale)

def ProductOrder.eval (f : ProductOrder) (a : Ops α) (x y : α) : α :=
  match f with | .direct => a.mul x y | .swapped => a.mul y x

def nativeCoefficient (a : Ops α) (c : CompilerLayout) (site : List Nat)
    (valid : Bool) (twice : α → α) (ds score : α) : α :=
  let product := (c.scalar.coefficientOrder site).eval a
  match c.scalar.coefficientMask site with
  | .beforeTwice => product (twice (if valid then ds else a.zero)) (if valid then score else a.zero)
  | .afterTwice => product (if valid then twice ds else a.zero) (if valid then score else a.zero)
  | .afterProduct => if valid then product (twice ds) score else a.zero

def nativeScale (a : Ops α) (c : CompilerLayout) (s : Shape) (site : List Nat) (x : α) : α :=
  if s.useScale then (c.scalar.scale site).eval a x (a.scale s.K true) else x

def nativeTwice (a : Ops α) (c : CompilerLayout) (site : List Nat) (x : α) : α :=
  (c.scalar.twice site).eval a x

def nativeScaleTwice (a : Ops α) (c : CompilerLayout) (s : Shape) (site : List Nat) (x : α) : α :=
  if s.useScale then (c.scalar.scaleTwice site).eval a x (a.scale s.K true)
  else nativeTwice a c site x

def nativeScaledSum (a : Ops α) (c : CompilerLayout) (s : Shape) (site : List Nat) (x z : α) : α :=
  (c.scalar.scaledSum site).eval a x z (a.scale s.K true)


end
end VeriTile.Bench.Repairs.ParallelAttention
