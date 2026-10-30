import bench.tritonbench_g.ksoftmax_triton.Proofs.Congruence

namespace VeriTile.Bench.Repairs.Ksoftmax.Graph
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

def zero : Word := BitVec.ofNat 32 0
def negInf : Word := BitVec.ofNat 32 0xff800000
def log2e : Word := BitVec.ofNat 32 0x3fb8aa3b

inductive Expr where
  | leaf (word : Word)
  | input (objectId index : Nat) (word : Word)
  | load (input : Expr)
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  | div (left right : Expr)
  | max (left right : Expr)
  | fma (left right addend : Expr)
  | neg (input : Expr)
  | ex2 (input : Expr)
  | log (input : Expr)
  deriving Inhabited

/-- Exact ordered native reduction syntax. Fused edges are explicit choices
of the compiled tree; they are never justified by an add/mul numerical law. -/
inductive Plan (ι : Type) where
  | lane (index : ι)
  | constant (word : Word)
  | add (left right : Plan ι)
  | max (left right : Plan ι)
  | fmaLeft (product other : Plan ι)
  | fmaRight (other product : Plan ι)

def fuseLeft : Expr → Expr → Expr
  | .mul x y, z => .fma x y z
  | x, y => .add x y
def fuseRight : Expr → Expr → Expr
  | z, .mul x y => .fma x y z
  | x, y => .add x y

def Plan.lower {ι : Type} (inputs : ι → Expr) : Plan ι → Expr
  | .lane i => inputs i
  | .constant w => .leaf w
  | .add a b => .add (a.lower inputs) (b.lower inputs)
  | .max a b => .max (a.lower inputs) (b.lower inputs)
  | .fmaLeft a b => fuseLeft (a.lower inputs) (b.lower inputs)
  | .fmaRight a b => fuseRight (a.lower inputs) (b.lower inputs)

/-- A local compiler reduction interface, instantiated at the actual
DEPTH/num_warps/layout. It supplies finite instruction trees, not output words
or a kernel execution equation. All six source configurations remain allowed.
The recorded PTX determines fused edges and operand order for each artifact. -/
structure ReductionPlans where
  sum : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Plan (TileIndex shape)
  maximum : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Plan (TileIndex shape)

inductive FMAStyle where
  | negativeLeftFirst | negativeLeftSecond | negativeRightFirst | negativeRightSecond

def FMAStyle.lower (style : FMAStyle) (left right addend : Expr) : Expr :=
  match style with
  | .negativeLeftFirst => .fma (.neg left) right addend
  | .negativeLeftSecond => .fma right (.neg left) addend
  | .negativeRightFirst => .fma (.neg right) left addend
  | .negativeRightSecond => .fma left (.neg right) addend

def inputIndex : Expr → Nat
  | .input _ i _ => i
  | .load x => inputIndex x
  | _ => 0

structure Profile where
  tritonVersion : String
  target : String
  numWarps : Nat
  sourceWarps : numWarps∈[1,2,4,8,16,32]
  enableFPFusion : Bool
  fusion : enableFPFusion=Bool.true
  columns : Nat
  tileDepth : Nat
  /-- A finite choice of actual FMA instruction form for each lane. This is
  compiler scheduling metadata, independent of input words and outputs. -/
  subtractProduct : Nat → FMAStyle
  plans : ReductionPlans

/-- The ordinary selected-compilation binding is explicit at native public
entries. Reduction trees and the four-way per-lane FMA choices in Profile
are the local compiler interface; they never supply a task output word. -/
structure Binding (profile : Profile) (K D : Nat) : Prop where
  columns : profile.columns=K
  tileDepth : profile.tileDepth=D
  compilerVersion : profile.tritonVersion="3.7.1"
  target : profile.target="cuda:80:32"

/-- The source LOG backward subtraction is emitted as a ternary FMA with a
negated multiplicand. The selected lane schedule retains both the sign's
placement and operand order; K=1 plain backward also contracts its subtraction.
Plain final multiplication retains the observed native
operand order. Loads are explicit barriers, so no expression stored by an
earlier call can accidentally be contracted across a later tensor load. -/
def binary (profile : Profile) : Binary → Expr → Expr → Expr
  | .add, x, y => .add x y
  | .sub, g, .mul e s => (profile.subtractProduct (inputIndex g%profile.columns)).lower e s g
  | .sub, x, y => .sub x y
  | .mul, o, .sub g s => .mul (.sub g s) o
  | .mul, o, .fma x y z => .mul (.fma x y z) o
  | .mul, x, y => .mul x y
  | .div, x, y => .div x y
  | .max, x, y => .max x y
  | .pow, _, _ => .leaf zero

def algebra (rest : Algebra Expr) (profile : Profile) : Algebra Expr :=
  {rest with
    literal := fun _ _ _ => .leaf zero
    negInf := .leaf negInf
    fp32Load := Expr.load
    cast := fun _ _ _ x => x
    binary := fun _ _ op => binary profile op
    unary := fun _ op x => match op with
      | .exp => .ex2 (.mul x (.leaf log2e))
      | .log => .log x
      | _ => .leaf zero
    reduceSum := fun _ {_} ax keep inputs i => (profile.plans.sum ax keep i).lower inputs
    reduceMax := fun _ {_} ax keep inputs i => (profile.plans.maximum ax keep i).lower inputs}

structure Operations where
  add32 : Word → Word → Word
  sub32 : Word → Word → Word
  mul32 : Word → Word → Word
  divFull32 : Word → Word → Word
  max32 : Word → Word → Word
  fmaRN32 : Word → Word → Word → Word
  ex2Approx32 : Word → Word
  libdeviceLog32 : Word → Word

def eval (ops : Operations) : Expr → Word
  | .leaf w => w
  | .input _ _ w => w
  | .load x => eval ops x
  | .add x y => ops.add32 (eval ops x) (eval ops y)
  | .sub x y => ops.sub32 (eval ops x) (eval ops y)
  | .mul x y => ops.mul32 (eval ops x) (eval ops y)
  | .div x y => ops.divFull32 (eval ops x) (eval ops y)
  | .max x y => ops.max32 (eval ops x) (eval ops y)
  | .fma x y z => ops.fmaRN32 (eval ops x) (eval ops y) (eval ops z)
  | .neg x => eval ops x ^^^ BitVec.ofNat 32 (2^31)
  | .ex2 x => ops.ex2Approx32 (eval ops x)
  | .log x => ops.libdeviceLog32 (eval ops x)

def reference (rest : Algebra Expr) (profile : Profile) (ops : Operations) (op : Operation)
    (Q K batch row leftId rightId : Nat) {D : Nat} (left right : Nat → Word) (i : Fin D) : Word :=
  eval ops ((gridValue (algebra rest profile) op Q K batch row
    (fun j => .input leftId j (left j)) (fun j => .input rightId j (right j)) i).read .real)

#specNonCircular reference avoiding [forward,backward,runRows]
#axiomsClean reference
end
end VeriTile.Bench.Repairs.Ksoftmax.Graph
