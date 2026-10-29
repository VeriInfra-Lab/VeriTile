import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedDivision

/-! The selected native graph is constructed with staged expressions, not
with separately rounded FP32 arithmetic. The frozen default sm90 profiles
contract both online updates, reverse the second score-scale multiplication,
fold the two initial constant additions, and erase same-native-dtype casts.
These are explicit graph choices, never floating-point equality assumptions.
The FP16-result value dot and its widening remain separate from the FMA. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open TypedExecution
noncomputable section
set_option maxHeartbeats 2000000

abbrev Word := BitVec 32

inductive Expr where
  | leaf (word : Word)
  | zero | one | negInf
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | fromInt (precision : Option ComputeDType) (value : Int)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | product (precision : Option ComputeDType) (dtype : FloatDType) (reversedFactors : Bool) (x y : Expr)
  | fma (precision : Option ComputeDType) (dtype : FloatDType) (x y acc : Expr)
  | reduction (precision : Option ComputeDType) (isMax : Bool)
      (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
      (inputs : TileIndex shape → Expr) (i : TileIndex (TileShape.reduceShape shape ax keep))
  | scoreDot (precision : Option ComputeDType) (inputPrecision : String)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch ++ [m,k]) → Expr) (b : TileIndex (batch ++ [k,n]) → Expr)
      (i : TileIndex (batch ++ [m,n]))
  | typedDot (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch ++ [m,k]) → Expr) (b : TileIndex (batch ++ [k,n]) → Expr)
      (i : TileIndex (batch ++ [m,n]))

instance : Inhabited Expr := ⟨.leaf 0⟩

def nativeDType : FloatDType → FloatDType | .real => .fp32 | d => d

def literal (p : Option ComputeDType) (d : FloatDType) (x : ℝ) : Expr := by
  classical
  exact if x=0 then .zero else if x=1 then .one else .literal p (nativeDType d) x

/-- A nested surviving product is the second score-scale multiplication.
Its native operands are K-scale, then the first scaled score. Products consumed
by an online-update FMA retain old-value, alpha, addend in that order. -/
def consume (p : Option ComputeDType) (d : FloatDType) : Binary → Expr → Expr → Expr
  | .mul,a@(.product ..),b => .product p (nativeDType d) Bool.true a b
  | .mul,a,b => .product p (nativeDType d) Bool.false a b
  | .add,.zero,.one => .one
  | .add,.zero,.negInf => .negInf
  | .add,.product _ _ _ a b,acc => .fma p (nativeDType d) a b acc
  | op,a,b => .binary p (nativeDType d) op a b

def cast (p : Option ComputeDType) (source target : FloatDType) (x : Expr) : Expr :=
  if nativeDType source=nativeDType target then x
  else .cast p (nativeDType source) (nativeDType target) x

/-- This Algebra constructs a graph. It does not interpret multiplication as
a rounded result before a subsequent addition consumes the product node. -/
def backend : Algebra Expr where
  literal := literal
  negInf := .negInf
  binary := consume
  unary := Expr.unary
  cast := cast
  fromNat := Expr.fromNat
  fromInt := Expr.fromInt
  fp32Bits := fun w => .leaf w.bits
  fp32Load := id
  reduceMax := fun p {shape} ax keep xs i => .reduction p Bool.true shape ax keep xs i
  reduceSum := fun p {shape} ax keep xs i => .reduction p Bool.false shape ax keep xs i
  dot := fun p {batch} {m k n} a b => some (fun i => .scoreDot p "tf32" batch m k n a b i)
  dotTyped := fun p left right out options {batch} {m k n} a b =>
    some (fun i => ofFloat out (.typedDot p (nativeDType left) (nativeDType right) (nativeDType out)
      options batch m k n (fun j => toFloat left (a j)) (fun j => toFloat right (b j)) i))

def support (BD : Nat) : DotSupport backend BD where
  score := fun a b i => .scoreDot (some .fp32) "tf32" [] 128 BD 64 a b i
  score_available := fun _ _ => rfl
  value := fun a b i => .typedDot (some .fp32) .fp16 .fp16 .fp16 {outDType:=some .fp16}
    [] 128 64 BD a b i
  value_available := fun _ _ => rfl

/-- Ordinary local instruction semantics. Reduction and dot retain their full
ordered input tiles and configuration. FMA is a distinct ternary operation;
no split-product/add or DotAcc implementation is substituted. Words in FP16
and BF16 typed cells use the selected backend's encoding in the low 16 bits. -/
structure Operations where
  scalar : Algebra Word
  fma : Option ComputeDType → FloatDType → Word → Word → Word → Word
  scoreDot : Option ComputeDType → String → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch ++ [m,k]) → Word) → (TileIndex (batch ++ [k,n]) → Word) →
    TileIndex (batch ++ [m,n]) → Word
  typedDot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch ++ [m,k]) → Word) → (TileIndex (batch ++ [k,n]) → Word) →
    TileIndex (batch ++ [m,n]) → Word

def evaluate (ops : Operations) : Expr → Word
  | .leaf word => word
  | .zero => 0
  | .one => 0x3f800000
  | .negInf => 0xff800000
  | .literal p d x => ops.scalar.literal p d x
  | .fromNat p x => ops.scalar.fromNat p x
  | .fromInt p x => ops.scalar.fromInt p x
  | .cast p source target x => ops.scalar.cast p source target (evaluate ops x)
  | .unary p op x => ops.scalar.unary p op (evaluate ops x)
  | .binary p d op x y => ops.scalar.binary p d op (evaluate ops x) (evaluate ops y)
  | .product p d reversedFactors x y =>
      if reversedFactors then ops.scalar.binary p d .mul (evaluate ops y) (evaluate ops x)
      else ops.scalar.binary p d .mul (evaluate ops x) (evaluate ops y)
  | .fma p d x y acc => ops.fma p d (evaluate ops x) (evaluate ops y) (evaluate ops acc)
  | .reduction p isMax shape ax keep xs i =>
      if isMax then ops.scalar.reduceMax p ax keep (fun j => evaluate ops (xs j)) i
      else ops.scalar.reduceSum p ax keep (fun j => evaluate ops (xs j)) i
  | .scoreDot p inputPrecision batch m k n a b i =>
      ops.scoreDot p inputPrecision batch m k n
        (fun j => evaluate ops (a j)) (fun j => evaluate ops (b j)) i
  | .typedDot p left right out options batch m k n a b i =>
      ops.typedDot p left right out options batch m k n
        (fun j => evaluate ops (a j)) (fun j => evaluate ops (b j)) i

theorem online_update (p : Option ComputeDType) (old alpha addend : Expr) :
    consume p .real .add (consume p .real .mul old alpha) addend =
      .fma p .fp32 old alpha addend := by cases old <;> rfl

theorem online_update_evaluated (ops : Operations) (p : Option ComputeDType)
    (old alpha addend : Expr) :
    evaluate ops (consume p .real .add (consume p .real .mul old alpha) addend) =
      ops.fma p .fp32 (evaluate ops old) (evaluate ops alpha) (evaluate ops addend) := by
  rw [online_update]
  rfl

theorem score_scale_order (p : Option ComputeDType) (score qs ks : Expr) :
    consume p .real .mul (consume p .real .mul score qs) ks =
      .product p .fp32 Bool.true (consume p .real .mul score qs) ks := by
  cases score <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.NativeGraph
