import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Execution

namespace VeriTile.Bench.Repairs.FlashDecode2.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev Word := BitVec 32

/-- Numerical words occur only in leaves. Compiler choices inspect Shape. -/
inductive Shape where
  | leaf (region : RegionName) (address : Nat)
  | literal (precision : Option ComputeDType) (dtype : FloatDType)
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
  | .literal p d _ => .literal p d
  | .negInf => .negInf
  | .binary p d op a b => .binary p d op (shape a) (shape b)
  | .unary p op a => .unary p op (shape a)
  | .cast p src dst a => .cast p src dst (shape a)
  | .fromNat p n => .fromNat p n
  | .fromInt p n => .fromInt p n
  | .bits w => .bits w
  | .load a => .load (shape a)

def algebra : Algebra Term where
  literal := .literal
  negInf := .negInf
  binary := .binary
  unary := .unary
  cast := .cast
  fromNat := .fromNat
  fromInt := .fromInt
  fp32Bits := fun w => .bits w.bits
  fp32Load := .load
  reduceMax := fun p _ _ _ _ _ => .literal p .real 0
  reduceSum := fun p _ _ _ _ _ => .literal p .real 0

/-- The source has no reduction instruction: its block loop is explicit.
Only individual numerical instructions/conversions are deferred. No
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
  | split | left | right
  deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion
  /-- Operand ordering chosen by the actual compiler, without a commutativity law. -/
  swap : Shape → Bool

/-- Preserve the compiler's operand order for source commutative nodes without
assuming equality of the two native orders (including NaN payload behavior). -/
def binary (I : Instructions) (S : Schedule) (p : Option ComputeDType) (d : FloatDType)
    (op : Binary) (a b : Term) (x y : Word) : Word :=
  if S.swap (.binary p d op (shape a) (shape b)) &&
      (op==.add || op==.mul || op==.max) then I.binary p d op y x else I.binary p d op x y

def fused (I : Instructions) (S : Schedule) (a b : Term) (x y z : Word) : Word :=
  if S.swap (.binary (some .fp32) .real .mul (shape a) (shape b)) then I.fmaRN y x z
  else I.fmaRN x y z

/-- Only actual adjacent FP32 mul/add may contract. Cast/load nodes remain
boundaries. The selection cannot inspect input words or supply a result. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => I.literal p d x
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | _,_,_,a,b => binary I S p d .add a b (eval I S a) (eval I S b)
  | .binary p d op a b => binary I S p d op a b (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast p src dst a => I.cast p src dst (eval I S a)
  | .fromNat p n => I.fromNat p n
  | .fromInt p n => I.fromInt p n
  | .bits w => w
  | .load a => eval I S a

/-- Input words are erased before either compiler selector sees the graph. -/
theorem leaf_shape (r : RegionName) (o : Nat) (u v : Word) :
    shape (.leaf r o u)=shape (.leaf r o v) := rfl

theorem literal_shape (p : Option ComputeDType) (d : FloatDType) (x y : ℝ) :
    shape (.literal p d x)=shape (.literal p d y) := rfl

def observe (I : Instructions) (S : Schedule) (s : State Term) (r : RegionName) (o : Nat) :=
  eval I S ((s.mem r o).read .real)

#axiomsClean leaf_shape
#axiomsClean literal_shape
#specNonCircular eval avoiding [FP.Structural.exec,kernel]
end
end VeriTile.Bench.Repairs.FlashDecode2.Native
