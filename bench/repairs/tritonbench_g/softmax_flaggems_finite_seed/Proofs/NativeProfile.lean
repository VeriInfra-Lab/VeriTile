import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.NativeHost
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Words
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.NativeProfile
open VeriTile.Triton FP.Structural
open Native
namespace G
abbrev Expr := Ksoftmax.Graph.Expr
abbrev Operations := Ksoftmax.Graph.Operations
abbrev eval := Ksoftmax.Graph.eval
end G
noncomputable section

def binary (op : Binary) (a b : G.Expr) : G.Expr := match op with
  | .add => .add a b | .sub => .sub a b | .mul => .mul a b
  | .div => .div a b | .max => .max a b | .pow => .leaf 0

def instructions (ops : G.Operations) (scalar : ℝ → Word) : Instructions where
  literal := fun _ _ x => if x=0 then 0 else if x=(-340282346638528859811704183484516925440.0 : ℝ) then 0xff7fffff else scalar x
  binary := fun _ _ op a b => G.eval ops (binary op (.leaf a) (.leaf b))
  unary := fun _ op a => if op=.exp then ops.ex2Approx32 (ops.mul32 a 0x3fb8aa3b) else 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => scalar n
  fromInt := fun _ n => scalar n
  fmaRN := ops.fmaRN32

/-- Explicit FP32 exp lowering and selected instruction graph. Source FP32
casts are identity. Zero and the finite seed have fixed FP32 encodings. -/
def lower (scalar : ℝ → Word) (S : Schedule) : Term → G.Expr
  | .leaf _ _ w => .leaf w
  | .literal _ _ x => .leaf (if x=0 then 0 else if x=(-340282346638528859811704183484516925440.0 : ℝ) then 0xff7fffff else scalar x)
  | .negInf => .leaf 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      .fma (lower scalar S x) (lower scalar S y) (lower scalar S c)
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (lower scalar S x) (lower scalar S y) (lower scalar S c)
    | .leftSwap,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      .fma (lower scalar S y) (lower scalar S x) (lower scalar S c)
    | .rightSwap,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (lower scalar S y) (lower scalar S x) (lower scalar S c)
    | _,_,_,a,b => .add (lower scalar S a) (lower scalar S b)
  | .binary p d .sub a b =>
    match S.fusion (.binary p d .sub (shape a) (shape b)),p,d,a,b with
    | .negativeLeftFirst,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (.neg (lower scalar S x)) (lower scalar S y) (lower scalar S c)
    | .negativeLeftSecond,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (lower scalar S y) (.neg (lower scalar S x)) (lower scalar S c)
    | .negativeRightFirst,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (.neg (lower scalar S y)) (lower scalar S x) (lower scalar S c)
    | .negativeRightSecond,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      .fma (lower scalar S x) (.neg (lower scalar S y)) (lower scalar S c)
    | _,_,_,a,b => .sub (lower scalar S a) (lower scalar S b)
  | .binary p d .mul a b =>
    if S.swapMultiply (.binary p d .mul (shape a) (shape b)) then .mul (lower scalar S b) (lower scalar S a)
    else .mul (lower scalar S a) (lower scalar S b)
  | .binary _ _ op a b => binary op (lower scalar S a) (lower scalar S b)
  | .unary _ op a => if op=.exp then .ex2 (.mul (lower scalar S a) (.leaf 0x3fb8aa3b)) else .leaf 0
  | .cast _ _ _ a => lower scalar S a
  | .fromNat _ n => .leaf (scalar n)
  | .fromInt _ n => .leaf (scalar n)
  | .bits w => .leaf w
  | .load a => .load (lower scalar S a)

theorem lower_exact (ops : G.Operations) (scalar : ℝ → Word) (S : Schedule) (t : Term) :
    Native.eval (instructions ops scalar) S t=G.eval ops (lower scalar S t) := by
  fun_induction Native.eval (instructions ops scalar) S t
  all_goals simp_all [Native.eval,lower,instructions,binary,G.eval,Ksoftmax.Graph.eval]
  all_goals split <;> simp_all [Ksoftmax.Graph.eval]

abbrev LocalExact := Ksoftmax.Words.LocalExact
abbrev NumericDomain := Ksoftmax.Words.NumericDomain

theorem exact_if_local (ops : G.Operations) (scalar : ℝ → Word) (S : Schedule)
    (ref : Ksoftmax.Words.PrimitiveReference) (ports : LocalExact ops ref) (t : Term)
    (domain : NumericDomain ops (lower scalar S t)) :
    Native.eval (instructions ops scalar) S t=
      G.eval (Ksoftmax.Words.referenceOps ref) (lower scalar S t) :=
  (lower_exact ops scalar S t).trans (Ksoftmax.Words.local_exact ops ref ports _ domain)

#specNonCircular lower avoiding [FPHost.invoke,FPGrid.run,FP.Structural.exec]
#axiomsClean lower_exact
#axiomsClean exact_if_local
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.NativeProfile
