import bench.repairs.tritonbench_g.rope_transform.Proofs.Row
namespace VeriTile.Bench.Repairs.RopeTransform.Native
open VeriTile Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

inductive Shape where
  | leaf (region : RegionName) (address : Nat)
  | literal (precision : Option ComputeDType) (dtype : FloatDType)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Shape)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Shape)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Shape)
  | fixedBits | unsupported
  deriving Inhabited
inductive Term where
  | leaf (region : RegionName) (address : Nat) (word : Word)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Term)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Term)
  | cast (precision : Option ComputeDType) (src dst : FloatDType) (a : Term)
  | fixedBits (value : Word)
  | unsupported
  deriving Inhabited

def shape : Term → Shape
  | .leaf r o _ => .leaf r o
  | .literal p d _ => .literal p d
  | .binary p d op a b => .binary p d op (shape a) (shape b)
  | .unary p op a => .unary p op (shape a)
  | .cast p src dst a => .cast p src dst (shape a)
  | .fixedBits _ => .fixedBits
  | .unsupported => .unsupported

/-- No reduction, tensor or whole-output oracle is present. The source has no
reductions; unsupported unrelated algebra operations have a fixed syntax node. -/
def algebra : Algebra Term where
  literal := .literal
  negInf := .unsupported
  binary := .binary
  unary := .unary
  cast := .cast
  fromNat := fun p n => .literal p .fp32 n
  fromInt := fun p n => .literal p .fp32 n
  fp32Bits := fun w => .fixedBits w.bits
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .unsupported
  reduceSum := fun _ {_} _ _ _ _ => .unsupported

/-- Only local scalar instruction interpretation is deferred. -/
structure Instructions where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  fmaRN : Word → Word → Word → Word
inductive Fusion where
  | split | left | right
  deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion
  swap : Shape → Bool

def binary (I : Instructions) (S : Schedule) (p : Option ComputeDType) (d : FloatDType)
    (op : Binary) (a b : Term) (x y : Word) :=
  if S.swap (.binary p d op (shape a) (shape b)) && (op==.add || op==.mul) then
    I.binary p d op y x else I.binary p d op x y

def fused (I : Instructions) (S : Schedule) (a b : Term) (x y z : Word) :=
  if S.swap (.binary none .fp32 .mul (shape a) (shape b)) then I.fmaRN y x z else I.fmaRN x y z

/-- Adjacent source mul/add or mul/sub nodes may contract only with the actual
none/FP32 tags. Casts are retained as boundaries. Subtraction retains sign
placement; no commutativity, reassociation or native=Real equality is used. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => I.literal p d x
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,none,.fp32,.binary none .fp32 .mul x y,c =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | .right,none,.fp32,c,.binary none .fp32 .mul x y =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | _,_,_,a,b => binary I S p d .add a b (eval I S a) (eval I S b)
  | .binary p d .sub a b =>
    match S.fusion (.binary p d .sub (shape a) (shape b)),p,d,a,b with
    | .left,none,.fp32,.binary none .fp32 .mul x y,c =>
      fused I S x y (eval I S x) (eval I S y) (I.unary none (.neg .fp32) (eval I S c))
    | .right,none,.fp32,c,.binary none .fp32 .mul x y =>
      fused I S x y (I.unary none (.neg .fp32) (eval I S x)) (eval I S y) (eval I S c)
    | _,_,_,a,b => I.binary p d .sub (eval I S a) (eval I S b)
  | .binary p d op a b => binary I S p d op a b (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast p src dst a => I.cast p src dst (eval I S a)
  | .fixedBits w => w
  | .unsupported => 0

theorem no_input_words_in_selector (r : RegionName) (o : Nat) (x y : Word) :
    shape (.leaf r o x)=shape (.leaf r o y) := rfl
theorem no_literal_values_in_selector (p : Option ComputeDType) (d : FloatDType) (x y : ℝ) :
    shape (.literal p d x)=shape (.literal p d y) := rfl

/-- Direct scalar graph on four input words, before schedule interpretation.
It contains no execution, memory update or final-result callback. -/
def graph (x y c sn : Term) (second : Bool) : Term :=
  let a := Term.cast none .fp32 .fp32 x
  let b := Term.cast none .fp32 .fp32 y
  .cast none .fp32 .fp32
    (if second then .binary none .fp32 .add (.binary none .fp32 .mul b c) (.binary none .fp32 .mul a sn)
     else .binary none .fp32 .sub (.binary none .fp32 .mul a c) (.binary none .fp32 .mul b sn))

theorem source_scalar (x y c sn : Term) (second : Bool) :
    scalar algebra x y c sn second=graph x y c sn second := rfl

#axiomsClean source_scalar
#axiomsClean no_input_words_in_selector
#axiomsClean no_literal_values_in_selector
#specNonCircular eval avoiding [kernel,FP.Structural.exec,T.referenceMemory]
#specNonCircular graph avoiding [kernel,FP.Structural.exec,T.referenceMemory,T.laneResult]
end
end VeriTile.Bench.Repairs.RopeTransform.Native
