import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelGraph
import bench.repairs.tritonbench_g.rms_matmul_rbe.Proofs.Selected
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelRealArithmetic

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32
abbrev DotPlan := RmsMatmulRbe.DotPlan
abbrev Tree := SoftmaxFlaggems.Native.Tree
abbrev Plan := SoftmaxFlaggems.Native.Plan
abbrev Fusion := SoftmaxFlaggems.Native.Fusion
local instance (p : Prop) : Decidable p := Classical.propDecidable p

/-- Shapes erase all runtime floating words, raw E5 bytes and signed-position
payloads. Literal constants are actual compile-time EPS/THETA/source constants. -/
inductive Shape where
  | leaf (region : RegionName) (address : Nat)
  | literal (p : Option ComputeDType) (d : FloatDType) (v : ℝ)
  | binary (p : Option ComputeDType) (d : FloatDType) (op : Binary) (a b : Shape)
  | unary (p : Option ComputeDType) (op : Unary) (a : Shape)
  | cast (p : Option ComputeDType) (src dst : FloatDType) (a : Shape)
  | fromNat (p : Option ComputeDType) (n : Nat)
  | fromInt (p : Option ComputeDType)
  | decode (d : FloatDType)
  | power (p : Option ComputeDType) (n : Int) (x : Shape)
  | dot (x y : Fin 64 → Shape)
  | load (x : Shape)
  deriving Inhabited
inductive Term where
  | leaf (region : RegionName) (address : Nat) (word : Word)
  | literal (p : Option ComputeDType) (d : FloatDType) (v : ℝ)
  | binary (p : Option ComputeDType) (d : FloatDType) (op : Binary) (a b : Term)
  | unary (p : Option ComputeDType) (op : Unary) (a : Term)
  | cast (p : Option ComputeDType) (src dst : FloatDType) (a : Term)
  | fromNat (p : Option ComputeDType) (n : Nat)
  | fromInt (p : Option ComputeDType) (n : Int)
  | decode (d : FloatDType) (byte : Float8E5Bits)
  | power (p : Option ComputeDType) (n : Int) (x : Term)
  | dot (plan : DotPlan) (x y : Fin 64 → Term)
  | load (x : Term)
  deriving Inhabited

def shape : Term → Shape
  | .leaf r a _ => .leaf r a
  | .literal p d v => .literal p d v
  | .binary p d op a b => .binary p d op (shape a) (shape b)
  | .unary p op a => .unary p op (shape a)
  | .cast p src dst a => .cast p src dst (shape a)
  | .fromNat p n => .fromNat p n
  | .fromInt p _ => .fromInt p
  | .decode d _ => .decode d
  | .power p n x => .power p n (shape x)
  | .dot _ x y => .dot (fun i => shape (x i)) (fun i => shape (y i))
  | .load x => .load (shape x)

/-- Only scalar instructions, scalar format conversions and a single fixed
m16n8k8 TF32 MMA output lane are numerical ports. -/
structure Instructions where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  decode : FloatDType → Float8E5Bits → Word
  power : Option ComputeDType → Word → Int → Word
  fmaRN : Word → Word → Word → Word
  mulFTZ : Word → Word → Word
  mma8 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

structure Schedule where
  fusion : Shape → Fusion
  swapMultiply : Shape → Bool
  swapAdd : Shape → Bool
  eraseFP32Cast : Shape → Bool
  /-- Actual compiler absorption of an adjacent external carry into the dot.
  Its value is never justified by a split/fused numerical equality. -/
  dotCarry : Shape → Bool

def dotWords (I : Instructions) (plan : DotPlan) (x y : Fin 64 → Word) (seed : Word) : Word :=
  plan.instructions.foldl (fun old lanes => I.mma8 (fun i => x (lanes i))
    (fun i => y (lanes i)) old) seed

/-- Bottom-up instruction interpretation keeps only immediate multiplication and
dot candidates for an adjacent parent contraction. Casts end those candidates. -/
structure Evaluated where
  word : Word
  product : Option (Word × Word) := none
  dot : Option (Word → Word) := none

def addWords (I : Instructions) (S : Schedule) (p : Option ComputeDType)
    (d : FloatDType) (node : Shape) (a b : Evaluated) : Word :=
  match p,d,b.dot with
  | some .fp32,.real,some dot =>
    if S.dotCarry node then dot a.word else I.binary p d .add a.word b.word
  | _,_,_ =>
    match S.fusion node,p,d,a.product,b.product with
    | .left,some .fp32,.real,some (x,y),_ => I.fmaRN x y b.word
    | .leftSwap,some .fp32,.real,some (x,y),_ => I.fmaRN y x b.word
    | .right,some .fp32,.real,_,some (x,y) => I.fmaRN x y a.word
    | .rightSwap,some .fp32,.real,_,some (x,y) => I.fmaRN y x a.word
    | _,_,_,_,_ =>
      if S.swapAdd node then I.binary p d .add b.word a.word
      else I.binary p d .add a.word b.word

def subWords (I : Instructions) (S : Schedule) (p : Option ComputeDType)
    (d : FloatDType) (node : Shape) (a b : Evaluated) : Word :=
  match S.fusion node,p,d,a.product,b.product with
  | .left,some .fp32,.real,some (x,y),_ => I.fmaRN x y (b.word ^^^ 0x80000000)
  | .leftSwap,some .fp32,.real,some (x,y),_ => I.fmaRN y x (b.word ^^^ 0x80000000)
  | .negativeLeftFirst,some .fp32,.real,_,some (x,y) => I.fmaRN (x ^^^ 0x80000000) y a.word
  | .negativeLeftSecond,some .fp32,.real,_,some (x,y) => I.fmaRN y (x ^^^ 0x80000000) a.word
  | .negativeRightFirst,some .fp32,.real,_,some (x,y) => I.fmaRN (y ^^^ 0x80000000) x a.word
  | .negativeRightSecond,some .fp32,.real,_,some (x,y) => I.fmaRN x (y ^^^ 0x80000000) a.word
  | _,_,_,_,_ => I.binary p d .sub a.word b.word

def evalData (I : Instructions) (S : Schedule) : Term → Evaluated
  | .leaf _ _ w => ⟨w,none,none⟩
  | .literal p d x => ⟨I.literal p d x,none,none⟩
  | .binary p d op a b =>
    let left := evalData I S a
    let right := evalData I S b
    let node := Shape.binary p d op (shape a) (shape b)
    match op with
    | .add => ⟨addWords I S p d node left right,none,none⟩
    | .sub => ⟨subWords I S p d node left right,none,none⟩
    | .mul =>
      let word := if S.swapMultiply node then I.binary p d .mul right.word left.word
        else I.binary p d .mul left.word right.word
      ⟨word,if p=some .fp32 ∧ d=.real then some (left.word,right.word) else none,none⟩
    | op => ⟨I.binary p d op left.word right.word,none,none⟩
  | .unary p op a => ⟨I.unary p op (evalData I S a).word,none,none⟩
  | .cast p src dst a =>
    let x := (evalData I S a).word
    let word := if S.eraseFP32Cast (.cast p src dst (shape a)) &&
        ((src==.real || src==.fp32) && (dst==.real || dst==.fp32)) then x
      else I.cast p src dst x
    ⟨word,none,none⟩
  | .fromNat p n => ⟨I.fromNat p n,none,none⟩
  | .fromInt p n => ⟨I.fromInt p n,none,none⟩
  | .decode d b => ⟨I.decode d b,none,none⟩
  | .power p n x =>
    let v := (evalData I S x).word
    ⟨if n=2 then I.mulFTZ (I.literal p .real 1) (I.mulFTZ v v) else I.power p v n,none,none⟩
  | .dot plan x y =>
    let dot := dotWords I plan (fun i => (evalData I S (x i)).word)
      (fun i => (evalData I S (y i)).word)
    ⟨dot (I.literal (some .fp32) .real 0),none,some dot⟩
  | .load x => ⟨(evalData I S x).word,none,none⟩

def eval (I : Instructions) (S : Schedule) (t : Term) : Word := (evalData I S t).word

abbrev Reductions := (isMax : Bool) → (p : Option ComputeDType) → (s : TileShape) →
  (ax : Fin s.length) → (keep : Bool) → (out : TileIndex (TileShape.reduceShape s ax keep)) →
  (Fin (TileShape.axisDim s ax) → Shape) → Plan (TileShape.axisDim s ax)
abbrev Dots := (Fin 64 → Shape) → (Fin 64 → Shape) → DotPlan

def expand (p : Option ComputeDType) (op : Binary) (xs : Fin n → Term) (neutral : Term) : Tree n → Term
  | .lane i => xs i
  | .neutral => neutral
  | .join a b => .binary p .real op (expand p op xs neutral a) (expand p op xs neutral b)
def reduction (R : Reductions) (isMax : Bool) (p : Option ComputeDType) {s : TileShape}
    (ax : Fin s.length) (keep : Bool) (xs : Values Term .real s)
    (out : TileIndex (TileShape.reduceShape s ax keep)) : Term :=
  let leaves := match keep with
    | Bool.false => fun k => xs (TileShape.insertAxisIndex s ax out k)
    | Bool.true => fun k => xs (TileShape.replaceAxisIndex s ax out k)
  expand p (if isMax then .max else .add) leaves (.literal p .real 0)
    (R isMax p s ax keep out (fun i => shape (leaves i))).tree

def dot (D : Dots) {k : Nat} (x y : Fin k → Term) : Term :=
  if hk : k=64 then
    let left := fun i : Fin 64 => x (Fin.cast hk.symm i)
    let right := fun i : Fin 64 => y (Fin.cast hk.symm i)
    .dot (D (fun i => shape (left i)) (fun i => shape (right i))) left right
  else (List.finRange k).foldl (fun old i => .binary (some .fp32) .real .add old
    (.binary (some .fp32) .real .mul (x i) (y i))) (.literal (some .fp32) .real 0)

def operations (R : Reductions) (D : Dots) : KernelSource.Operations Term where
  scalar := {
    literal := .literal
    negInf := .leaf "negative infinity" 0 0xff800000
    binary := .binary
    unary := .unary
    cast := .cast
    fromNat := .fromNat
    fromInt := .fromInt
    fp32Bits := fun w => .leaf "constant bits" 0 w.bits
    fp32Load := .load
    reduceMax := fun p => reduction R Bool.true p
    reduceSum := fun p => reduction R Bool.false p }
  power := fun p x n => .power p n x
  decode := .decode
  dot := fun _ _ _ _ _ => TritonLinearActivation.KernelRealArithmetic.laneDot (dot D)

def reference (I : Instructions) (S : Schedule) (R : Reductions) (D : Dots)
    (fp8 : Bool) (p : KernelSource.Parameters) (v : KernelGraph.Input Term)
    (t : KernelGraph.Tile) (i : TileIndex [16,64]) : Word :=
  eval I S (KernelGraph.output (operations R D) fp8 p v t i)

theorem runtime_position_shape (p : Option ComputeDType) (x y : Int) :
    shape (.fromInt p x)=shape (.fromInt p y) := rfl
theorem raw_byte_shape (d : FloatDType) (x y : Float8E5Bits) :
    shape (.decode d x)=shape (.decode d y) := rfl

#specNonCircular reference avoiding [FP.Structural.exec, KernelSource.update, KernelSource.accumulated,
  KernelSource.rotated, KernelSource.scatter]
#axiomsClean runtime_position_shape
#axiomsClean raw_byte_shape
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNative
