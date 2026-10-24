import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraph

namespace VeriTile.Bench.Repairs.LayerNormWelfold.NativeKernel
open VeriTile.Triton FP.Structural
namespace Compiler
export VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
  (Term Shape Instructions Fusion shape)
end Compiler
abbrev Word := BitVec 32
abbrev Term := Compiler.Term
noncomputable section
local instance (p : Prop) : Decidable p := Classical.propDecidable p

/-- Individual native arithmetic/conversion ports including FMA.
No finite-input restriction or relationship to exact Real arithmetic is assumed. -/
abbrev Instructions := Compiler.Instructions

/-- Native addition and multiplication may each select their operand order.
The schedule sees operation shape and physical leaf identity, never raw input
words. In particular NaN payload order is not silently quotiented away. -/
structure Schedule where
  fusion : Compiler.Shape → Compiler.Fusion
  swapMultiply : Compiler.Shape → Bool
  swapAdd : Compiler.Shape → Bool
  dropZeroLeft : Compiler.Shape → Bool
  dropZeroRight : Compiler.Shape → Bool
  dropDivOne : Compiler.Shape → Bool
  eraseFP32Cast : Compiler.Shape → Bool

private def literalZero : Term → Bool
  | .literal _ _ value => decide (value = 0)
  | .fromNat _ 0 => Bool.true
  | _ => Bool.false

private def literalOne : Term → Bool
  | .literal _ _ value => decide (value = 1)
  | .fromNat _ 1 => Bool.true
  | _ => Bool.false

/-- Explicit scalar native lowering. Adjacent mul/add contraction emits a genuine
ternary instruction. Casts and physical leaves remain expression boundaries. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => I.literal p d x
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    if S.dropZeroLeft (.binary p d .add (Compiler.shape a) (Compiler.shape b)) && literalZero a then eval I S b
    else if S.dropZeroRight (.binary p d .add (Compiler.shape a) (Compiler.shape b)) && literalZero b then eval I S a
    else match S.fusion (.binary p d .add (Compiler.shape a) (Compiler.shape b)), p, d, a, b with
    | .left, some .fp32, .real, .binary (some .fp32) .real .mul x y, c =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .right, some .fp32, .real, c, .binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S x) (eval I S y) (eval I S c)
    | .leftSwap, some .fp32, .real, .binary (some .fp32) .real .mul x y, c =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | .rightSwap, some .fp32, .real, c, .binary (some .fp32) .real .mul x y =>
      I.fmaRN (eval I S y) (eval I S x) (eval I S c)
    | _, _, _, a, b =>
      if S.swapAdd (.binary p d .add (Compiler.shape a) (Compiler.shape b)) then
        I.binary p d .add (eval I S b) (eval I S a)
      else I.binary p d .add (eval I S a) (eval I S b)
  | .binary p d .mul a b =>
    if S.swapMultiply (.binary p d .mul (Compiler.shape a) (Compiler.shape b)) then
      I.binary p d .mul (eval I S b) (eval I S a)
    else I.binary p d .mul (eval I S a) (eval I S b)
  | .binary p d .div a b =>
    if S.dropDivOne (.binary p d .div (Compiler.shape a) (Compiler.shape b)) && literalOne b then eval I S a
    else I.binary p d .div (eval I S a) (eval I S b)
  | .binary p d op a b => I.binary p d op (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast (some .fp32) .real .real a =>
    if S.eraseFP32Cast (.cast (some .fp32) .real .real (Compiler.shape a)) then eval I S a
    else I.cast (some .fp32) .real .real (eval I S a)
  | .cast p src dst a => I.cast p src dst (eval I S a)
  | .fromNat p n => I.fromNat p n
  | .fromInt p n => I.fromInt p n
  | .bits w => w
  | .load a => eval I S a

/-- The compiler profile chooses contractions and operand ordering from operation
shape only. The selected default-fusion profile contracts adjacent FP32 mul/add sites; which multiplicand comes first remains explicit. -/
structure Profile where
  schedule : Schedule
  rightFused : ∀ a b c, schedule.fusion (.binary (some .fp32) .real .add c
    (.binary (some .fp32) .real .mul a b)) = .right ∨
    schedule.fusion (.binary (some .fp32) .real .add c
      (.binary (some .fp32) .real .mul a b)) = .rightSwap
  leftFused : ∀ a b c, (∀ x y, c ≠ .binary (some .fp32) .real .mul x y) →
    schedule.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) c) = .left ∨
    schedule.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) c) = .leftSwap

/-- A concrete inhabitant of the compiler-profile interface. It contracts
right-adjacent products first, otherwise left-adjacent products, preserving
source operand order and retaining explicit constants/casts. This witnesses
that the profile premises are consistent; it is not a compiler trace claim. -/
def sourceOrderProfile : Profile where
  schedule := {
    fusion := fun node => match node with
      | .binary (some .fp32) .real .add _ (.binary (some .fp32) .real .mul _ _) => .right
      | _ => .left
    swapMultiply := fun _ => Bool.false
    swapAdd := fun _ => Bool.false
    dropZeroLeft := fun _ => Bool.false
    dropZeroRight := fun _ => Bool.false
    dropDivOne := fun _ => Bool.false
    eraseFP32Cast := fun _ => Bool.false }
  rightFused := fun _ _ _ => .inl rfl
  leftFused := by
    intro a b c notMul
    left
    dsimp only
    split <;> simp_all


def observe (I : Instructions) (P : Profile) (t : Term) : Word :=
  eval I P.schedule t

/-- No whole-reduction callback is used: the explicit helper tree below expands
all its arithmetic. These unused reduction fields cannot supply any output. -/
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
  reduceMax := fun _ {_} _ _ _ => fun _ => .negInf
  reduceSum := fun p {_} _ _ _ => fun _ => .literal p .real 0

/-- Expression constructors, not numerical instruction implementations. -/
def constructors : Native.Primitives Term where
  zero := .literal (some .fp32) .real 0
  one := .fromNat (some .fp32) 1
  add := .binary (some .fp32) .real .add
  sub := .binary (some .fp32) .real .sub
  mul := .binary (some .fp32) .real .mul
  div := .binary (some .fp32) .real .div
  fma := fun a b c => .binary (some .fp32) .real .add (.binary (some .fp32) .real .mul a b) c
  unary := fun op a => match op with
    | 0 => .cast (some .fp32) .real .real a
    | 1 => .cast none .real .bf16 a
    | 2 => .unary (some .fp32) .rsqrt a
    | _ => a

theorem bindings : Native.SourceBindings algebra constructors := by
  constructor <;> intros <;> rfl

/-- Observe every floating payload while preserving exact cell dtype tags. -/
def observeValue (I : Instructions) (P : Profile) :
    (d : TileDType) → Value Term d → Value Word d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => observe I P
  | .nat | .int | .bool | .ptr | .blockPtr => id

def observeCell (I : Instructions) (P : Profile) : Cell Term → Cell Word
  | .mk d v => .mk d (observeValue I P d v)

structure Result where
  mean : Word
  inverse : Word
  output : Nat → Word

/-- Independent native target over logical leaves and the ordered two-pass
recurrence. It never reads source registers, source outputs or an execution. -/
def reference (I : Instructions) (P : Profile) (x w bias : Nat → Term)
    (D B row : Nat) (tree : Reuse.ReductionTree B) : Result :=
  let input := KernelGraph.logicalInputs algebra x w bias D row
  ⟨observe I P (Native.eval constructors input (Native.meanExpr D B tree)),
   observe I P (Native.eval constructors input (Native.inverseExpr D B tree)),
   fun col => observe I P (Native.eval constructors input (Native.outputExpr D B col tree))⟩

#specNonCircular eval avoiding [FP.Structural.exec]
#specNonCircular reference avoiding [FP.Structural.exec, meanValue, inverseValue, rowOutput]
#axiomsClean bindings
end
end VeriTile.Bench.Repairs.LayerNormWelfold.NativeKernel
