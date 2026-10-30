import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelReference
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Compiler

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelNative
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

/-- Primitive target semantics. These are model parameters, not numerical
transformation assumptions or deferred GPU experiments. FTZ/rounding/NaN
policies belong to these selected primitive definitions. -/
structure Instructions where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  fmaRN : Word → Word → Word → Word
  negate : Word → Word

inductive Fusion where
  | split | left | right
  deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion
  /-- Operand ordering chosen by the actual compiler, without a commutativity law. -/
  swap : Shape → Bool
  negateRight : Shape → Bool
  /-- Selected compiler specialization for a literal unit denominator. -/
  eraseUnitDivision : Bool
  foldConstantZeros : Bool

abbrev OutputSchedule := RegionName → Nat → Schedule

/-- Structural simplification of identity FP32 casts and selected constant
specializations, before contraction is inspected. No input-dependent rewrite
or numerical equivalence premise is used. -/
def normalize (S : Schedule) : Term → Term
  | .binary p d op a b =>
    let x := normalize S a
    let y := normalize S b
    match op,y with
    | .div,.literal q e one =>
      if S.eraseUnitDivision ∧ p=some .fp32 ∧ d=.real ∧ q=some .fp32 ∧ e=.real ∧ one=1
      then x else .binary p d op x y
    | .add,.literal q e zero =>
      match x with
      | .literal qx ex zx =>
        if S.foldConstantZeros ∧ p=some .fp32 ∧ d=.real ∧ q=some .fp32 ∧ e=.real ∧ zero=0 ∧
          qx=some .fp32 ∧ ex=.real ∧ zx=0 then .literal (some .fp32) .real 0
        else .binary p d op x y
      | _ => .binary p d op x y
    | _,_ => .binary p d op x y
  | .unary p op a => .unary p op (normalize S a)
  | .cast p src dst a =>
    if p=some .fp32 ∧ src=.real ∧ dst=.real then normalize S a else .cast p src dst (normalize S a)
  | .load a => normalize S a
  | t => t

/-- Preserve the compiler's operand order for source commutative nodes without
assuming equality of the two native orders (including NaN payload behavior). -/
def binary (I : Instructions) (S : Schedule) (p : Option ComputeDType) (d : FloatDType)
    (op : Binary) (a b : Term) (x y : Word) : Word :=
  if S.swap (.binary p d op (shape a) (shape b)) &&
      (op==.add || op==.mul || op==.max) then I.binary p d op y x else I.binary p d op x y

def fused (I : Instructions) (S : Schedule) (a b : Term) (x y z : Word) : Word :=
  if S.swap (.binary (some .fp32) .real .mul (shape a) (shape b)) then I.fmaRN y x z
  else I.fmaRN x y z

/-- Adjacent multiply/add/subtract contraction keeps the selected operand
order and negation position. No reassociation equality is postulated. -/
def eval (I : Instructions) (S : Schedule) : Term → Word
  | .leaf _ _ w => w
  | .literal p d x => if p=some .fp32 ∧ d=.real ∧ x=0 then 0 else I.literal p d x
  | .negInf => 0xff800000
  | .binary p d .add a b =>
    match S.fusion (.binary p d .add (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      fused I S x y (eval I S x) (eval I S y) (eval I S c)
    | _,_,_,a,b => binary I S p d .add a b (eval I S a) (eval I S b)
  | .binary p d .sub a b =>
    match S.fusion (.binary p d .sub (shape a) (shape b)),p,d,a,b with
    | .left,some .fp32,.real,.binary (some .fp32) .real .mul x y,c =>
      fused I S x y (eval I S x) (eval I S y) (I.negate (eval I S c))
    | .right,some .fp32,.real,c,.binary (some .fp32) .real .mul x y =>
      if S.negateRight (.binary p d .sub (shape a) (shape b)) then
        fused I S x y (eval I S x) (I.negate (eval I S y)) (eval I S c)
      else fused I S x y (I.negate (eval I S x)) (eval I S y) (eval I S c)
    | _,_,_,a,b => binary I S p d .sub a b (eval I S a) (eval I S b)
  | .binary p d op a b => binary I S p d op a b (eval I S a) (eval I S b)
  | .unary p op a => I.unary p op (eval I S a)
  | .cast p src dst a => if p=some .fp32 ∧ src=.real ∧ dst=.real then eval I S a
      else I.cast p src dst (eval I S a)
  | .fromNat p n => I.fromNat p n
  | .fromInt p n => I.fromInt p n
  | .bits w => w
  | .load a => eval I S a

def observe (I : Instructions) (S : Schedule) (cell : Cell Term) : Word :=
  eval I S (normalize S (cell.read .real))

theorem reduction_complete (plan : Plan n) :
    plan.tree.leaves.Perm ((List.finRange n).map some++List.replicate plan.padding none) := plan.valid

theorem leaf_shape (r : RegionName) (o : Nat) (u v : Word) :
    shape (.leaf r o u)=shape (.leaf r o v) := rfl

def sourceSHA256 := "ca518babd0de189bce565bce2c0456ae94883c32cd5b844f0044df994bd4a636"

/-- Structural object inspection. It consumes the actual bytes, ABI and
options, and cannot inspect numerical input words or return a kernel result. -/
structure Decoder where
  reductions : ByteArray → Compiler.Signature → Compiler.Options → Compiler.PrimitiveNames → Reductions
  schedule : ByteArray → Compiler.Signature → Compiler.Options → Compiler.PrimitiveNames → OutputSchedule

/-- Explicit compiler-model binding for the independent native graph. The
retained selected-object contract supplies source/ABI/FP32/mask/address facts;
the two extra equations are graph identity, never FP arithmetic laws. -/
structure Binding (platform : Compiler.Platform) (decoder : Decoder) (N B : Nat)
    (entry : Compiler.Entry) (R : Reductions) (S : OutputSchedule) where
  selected : Compiler.Selected platform.decoder N B
  entryMatch : selected.object.entry=entry
  pointers : selected.object.signature.pointers=List.replicate 6 32
  trees : decoder.reductions selected.object.bytes selected.object.signature
    selected.object.options selected.object.primitives=R
  order : decoder.schedule selected.object.bytes selected.object.signature
    selected.object.options selected.object.primitives=S

/-- Target scalar instruction semantics are selected by the actual object's
primitive identifiers. Their rounding/FTZ/NaN behavior is model data. -/
abbrev PrimitiveModel := Compiler.PrimitiveNames → Instructions

/-- The literal interpreter fixes positive zero independently of the target's
other conversion primitives. This is exactly the scalar leaf used by `eval`. -/
def literal32 (I : Instructions) (value : ℝ) : Word :=
  if value=0 then 0 else I.literal (some .fp32) .real value

/-- Preserve the original host scalar's binary64/integer/boolean ABI when
obtaining the actual FP32 parameter word of the selected object. -/
def epsilonWord (platform : Compiler.Platform) (object : Compiler.Object)
    (host : Compiler.Shared.HostScalar) : Word :=
  BitVec.ofNat 32 (platform.epsilon object host).toNat

/-- A single scalar-conversion model obligation. It mentions neither kernel
outputs nor execution: the source epsilon leaf denotes the selected ABI word.
In particular, positive host epsilon may convert to zero or infinity. -/
def EpsilonConversion (platform : Compiler.Platform) (object : Compiler.Object)
    (P : PrimitiveModel) (value : ℝ) (host : Compiler.Shared.HostScalar) : Prop :=
  literal32 (P object.primitives) value=epsilonWord platform object host

/-- Expose the runtime epsilon word in the independent native reference. The
conversion lemma below proves equality of the entire instruction record, so
even a coinciding count/zero/unit literal cannot acquire a different meaning.
The zero case retains the interpreter's fixed zero conversion. No graph shape,
normalization choice, operand order or FMA choice is changed. -/
def withEpsilon (I : Instructions) (value : ℝ) (word : Word) : Instructions :=
  { I with literal := fun p d x =>
      if p=some .fp32 ∧ d=.real ∧ x=value ∧ value≠0 then word else I.literal p d x }

theorem withEpsilon_eq (I : Instructions) (value : ℝ) (word : Word)
    (conversion : literal32 I value=word) : withEpsilon I value word=I := by
  have literals : (fun p d x =>
      if p=some .fp32 ∧ d=.real ∧ x=value ∧ value≠0 then word else I.literal p d x)=I.literal := by
    funext p d x
    split
    · rename_i h
      rcases h with ⟨rfl,rfl,rfl,hn⟩
      simpa only [literal32,if_neg hn] using conversion.symm
    · rfl
  unfold withEpsilon
  rw [literals]

theorem eval_literal32 (I : Instructions) (S : Schedule) (value : ℝ) :
    eval I S (.literal (some .fp32) .real value)=literal32 I value := by
  simp only [eval,literal32,true_and]

theorem bound_epsilon_operand (platform : Compiler.Platform) (object : Compiler.Object)
    (P : PrimitiveModel) (value : ℝ) (host : Compiler.Shared.HostScalar)
    (conversion : EpsilonConversion platform object P value host) (S : Schedule) :
    eval (withEpsilon (P object.primitives) value (epsilonWord platform object host)) S
      (.literal (some .fp32) .real value)=epsilonWord platform object host := by
  rw [withEpsilon_eq _ _ _ conversion,eval_literal32]
  exact conversion

theorem cast32_identity (I : Instructions) (S : Schedule) (a : Term) :
    eval I S (.cast (some .fp32) .real .real a)=eval I S a := by simp [eval]

#axiomsClean reduction_complete
#axiomsClean leaf_shape
#axiomsClean cast32_identity
#axiomsClean withEpsilon_eq
#axiomsClean bound_epsilon_operand
#specNonCircular EpsilonConversion avoiding [FP.Structural.exec,forwardKernel,backwardKernel,affineKernel]
#specNonCircular eval avoiding [FP.Structural.exec,forwardKernel,backwardKernel,affineKernel]
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelNative
