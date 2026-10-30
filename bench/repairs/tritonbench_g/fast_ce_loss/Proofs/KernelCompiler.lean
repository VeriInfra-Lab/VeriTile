import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelABI

/-! Task-local native lowering for same-FP32 casts and both subtraction
contraction orientations. The shared compiler graph stays byte-for-byte frozen. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.KernelCompiler
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
noncomputable section
set_option maxHeartbeats 4000000

/-- Same-FP32 conversions are bit-preserving compiler nodes. The original
source graph retains them; instruction selection sees their lowered graph.
No arithmetic reassociation or multiplication reordering occurs here. -/
def lower : Term → Term
  | .binary p d op a b => .binary p d op (lower a) (lower b)
  | .unary p op a => .unary p op (lower a)
  | .cast (some .fp32) .real .real a => lower a
  | .cast p src dst a => .cast p src dst (lower a)
  | .load a => .load (lower a)
  | a => a

/-- Adjacent operations contract only at shape-selected source sites.
`x*y-c` uses a negated addend; `c-x*y` negates one product operand. -/
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
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c ^^^ 0x80000000)
    | .leftSwap,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c ^^^ 0x80000000)
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

def observe (I : Instructions) (S : Schedule) (term : Term) : Word := eval I S (lower term)

theorem product_subtract_fused (I : Instructions) (S : Schedule) (x y c : Term)
    (selected : S.fusion (.binary (some .fp32) .real .sub
      (.binary (some .fp32) .real .mul (shape x) (shape y)) (shape c))=.left) :
    eval I S (.binary (some .fp32) .real .sub (.binary (some .fp32) .real .mul x y) c)=
      I.fmaRN (eval I S x) (eval I S y) (eval I S c ^^^ 0x80000000) := by
  simp [eval,shape,selected]

theorem subtract_product_fused (I : Instructions) (S : Schedule) (x y c : Term)
    (selected : S.fusion (.binary (some .fp32) .real .sub
      (shape c) (.binary (some .fp32) .real .mul (shape x) (shape y)))=.negativeLeftFirst) :
    eval I S (.binary (some .fp32) .real .sub c (.binary (some .fp32) .real .mul x y))=
      I.fmaRN (eval I S x ^^^ 0x80000000) (eval I S y) (eval I S c) := by
  simp [eval,shape,selected]

theorem same_fp32_cast (I : Instructions) (S : Schedule) (x : Term) :
    observe I S (.cast (some .fp32) .real .real x)=observe I S x := rfl

#axiomsClean product_subtract_fused
#axiomsClean subtract_product_fused
#axiomsClean same_fp32_cast
#specNonCircular observe avoiding [FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelCompiler
