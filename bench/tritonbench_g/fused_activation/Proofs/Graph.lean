import bench.tritonbench_g.fused_activation.Proofs.Grid
import VeriTile.Triton.Float.ScalarOps

/-! Exact native instruction graph observed in Triton 3.7.1, CUDA SM90,
default options. This stages a source multiply/add as FMA; it does not assert
that independently rounded multiplication/addition equals FMA. -/
namespace VeriTile.Bench.Repairs.FusedActivation.Graph
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

def zero : Word := BitVec.ofNat 32 0
def half : Word := BitVec.ofNat 32 0x3f000000
def one : Word := BitVec.ofNat 32 0x3f800000
def log2e : Word := BitVec.ofNat 32 0x3fb8aa3b

inductive Expr where
  | leaf (word : Word)
  | literalValue (value : ℝ)
  | product (left right : Expr)
  | fma (input multiplier x : Expr)
  | sigmoid (input : Expr)
  | maximum (input zero : Expr)
  | otherBinary (op : Binary) (left right : Expr)
  | otherUnary (op : Unary) (input : Expr)
  deriving Inhabited

def binary : Binary → Expr → Expr → Expr
  | .mul, m, input => .product m input
  | .add, .product m input, x => .fma input m x
  | .max, z, input => .maximum input z
  | op, x, y => .otherBinary op x y

def unary : Unary → Expr → Expr
  | .sigmoid, input => .sigmoid input
  | op, input => .otherUnary op input

def algebra (rest : Algebra Expr) : Algebra Expr :=
  { rest with
    literal := fun _ _ x => .literalValue x
    fp32Load := id
    cast := fun _ _ _ x => x
    binary := fun _ _ op => binary op
    unary := fun _ op => unary op }

/-- Instruction-sized interfaces, including exact native approximate operations.
These are not a callback for sigmoid or for the kernel's result. -/
structure Operations where
  scalar32 : ℝ → Word
  fmaRN32 : Word → Word → Word → Word
  add32 : Word → Word → Word
  sub32 : Word → Word → Word
  mul32 : Word → Word → Word
  ex2Approx32 : Word → Word
  divFull32 : Word → Word → Word
  max32 : Word → Word → Word
  otherBinary : Binary → Word → Word → Word
  otherUnary : Unary → Word → Word

/-- The actual five-instruction native sigmoid expansion (seven with FMA/add). The coefficient is
its encoded FP32 word, not the exact Real logarithm. No CR sigmoid claim. -/
def sigmoid (ops : Operations) (z : Word) : Word :=
  ops.divFull32 one (ops.add32 (ops.ex2Approx32 (ops.mul32 (ops.sub32 zero z) log2e)) one)

def eval (ops : Operations) : Expr → Word
  | .leaf w => w
  | .literalValue r => if r=0 then zero else if r=1/2 then half else ops.scalar32 r
  | .product x y => ops.mul32 (eval ops x) (eval ops y)
  | .fma input m x => ops.fmaRN32 (eval ops input) (eval ops m) (eval ops x)
  | .sigmoid z => sigmoid ops (eval ops z)
  | .maximum z v => ops.max32 (eval ops z) (eval ops v)
  | .otherBinary .add x y => ops.add32 (eval ops x) (eval ops y)
  | .otherBinary op x y => ops.otherBinary op (eval ops x) (eval ops y)
  | .otherUnary op x => ops.otherUnary op (eval ops x)

def nativeReference (ops : Operations) (activation : Bool) (x bias input : Word) : Word :=
  let z := ops.add32 (ops.fmaRN32 input half x) bias
  if activation then sigmoid ops z else ops.max32 z zero

/-- Raw leaves are essential: arbitrary loaded expressions could accidentally
match the compiler's multiplication/addition pattern across a load boundary. -/
theorem source_words (rest : Algebra Expr) (ops : Operations) (activation : Bool)
    (x bias input : Word) :
    eval ops (scalar (algebra rest) (1/2) activation (.leaf x) (.leaf bias) (.leaf input)) =
      nativeReference ops activation x bias input := by
  cases activation <;> simp [scalar,algebra,binary,unary,eval,nativeReference]

/-- Mathematical interpretation of this same source graph. It is independent
of native rounding and approximation, and makes no FP=Real assertion. -/
def realEval (leaf : Word → ℝ) : Expr → ℝ
  | .leaf w => leaf w
  | .literalValue r => r
  | .product x y => realEval leaf x * realEval leaf y
  | .fma input m x => realEval leaf input * realEval leaf m + realEval leaf x
  | .sigmoid z => Real.sigmoid (realEval leaf z)
  | .maximum z v => max (realEval leaf z) (realEval leaf v)
  | .otherBinary .add x y => realEval leaf x + realEval leaf y
  | .otherBinary _ _ _ => 0
  | .otherUnary _ _ => 0

def mathematical (activation : Bool) (x bias input : ℝ) : ℝ :=
  let z := ((1/2)*input+x)+bias
  if activation then Real.sigmoid z else max 0 z

theorem source_real (rest : Algebra Expr) (leaf : Word → ℝ) (activation : Bool)
    (x bias input : Word) :
    realEval leaf (scalar (algebra rest) (1/2) activation (.leaf x) (.leaf bias) (.leaf input)) =
      mathematical activation (leaf x) (leaf bias) (leaf input) := by
  cases activation <;> simp [scalar,algebra,binary,unary,realEval,mathematical,mul_comm,max_comm]

#specNonCircular nativeReference avoiding [candidateKernel,FP.Structural.exec,runGrid]
#specNonCircular mathematical avoiding [candidateKernel,FP.Structural.exec,runGrid,nativeReference]
#axiomsClean source_words
#axiomsClean source_real
end
end VeriTile.Bench.Repairs.FusedActivation.Graph
