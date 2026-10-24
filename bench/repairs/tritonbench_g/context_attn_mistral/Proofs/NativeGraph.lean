import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.FPHost
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeIndex

/-! Staged execution of the complete source. Products and dots are graph
payloads until their consumer chooses a native instruction. In particular the
normalizer's left product and add become one FMA, and both source dot/add sites
become dotAcc nodes with their actual accumulator. No equation between two
separately rounded operations and a fused instruction is assumed. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 2000000

/-- The ordered reduction tree is structural profile data. Leaves retain the
source ax index and padding remains explicit; no reassociation is applied. -/
inductive Tree (n : Nat) where
  | input (i : Fin n)
  | padding (word : BitVec 32)
  | node (a b : Tree n)

def Tree.indices : Tree n → List (Fin n)
  | .input i => [i] | .padding _ => [] | .node a b => a.indices++b.indices

structure OrderedTree (n : Nat) where
  tree : Tree n
  complete : tree.indices.Perm (List.finRange n)

inductive Expr where
  | leaf (dtype : FloatDType) (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | fromInt (precision : Option ComputeDType) (value : Int)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | product (precision : Option ComputeDType) (dtype : FloatDType)
      (reverse : Bool) (x y : Expr)
  | fma (precision : Option ComputeDType) (dtype : FloatDType) (x y acc : Expr)
  | pendingDot (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (acc : Expr) (i : TileIndex (batch++[m,n]))

instance : Inhabited Expr := ⟨.leaf .fp32 0⟩

def nativeDType : FloatDType → FloatDType
  | .real => .fp32 | d => d

/-- Only structural choices: ordered reduction trees and native operand
orientation. These fields cannot supply numerical values or kernel results.
The precise compiler profile and node provenance are bound externally. -/
structure Plan where
  reverseMul : Option ComputeDType → FloatDType → Expr → Expr → Bool
  reverseFMA : Option ComputeDType → FloatDType → Expr → Expr → Bool
  sumTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → OrderedTree (TileShape.axisDim shape ax)
  maxTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → OrderedTree (TileShape.axisDim shape ax)

/-- Only comparisons of two reached floating expressions choose a Boolean.
Its numerical implementation is local, never a program-success callback. -/
abbrev Equality := Option ComputeDType → FloatDType → Expr → Expr → Bool

def dotOptions (left right : FloatDType) (source : DotOptions) : DotOptions :=
  if nativeDType left = .fp32 ∧ nativeDType right = .fp32 ∧ source.inputPrecision = none then
    { source with inputPrecision := some "tf32" }
  else source

def consume (plan : Plan) (p : Option ComputeDType) (d : FloatDType) : Binary → Expr → Expr → Expr
  | .mul,a,b => .product p (nativeDType d) (plan.reverseMul p (nativeDType d) a b) a b
  | .add,acc,.pendingDot dp l r out opts batch m k n a b i =>
      .dotAcc dp l r out opts batch m k n a b acc i
  | .add,.product _ _ _ a b,acc =>
      if plan.reverseFMA p (nativeDType d) a b then .fma p (nativeDType d) b a acc
      else .fma p (nativeDType d) a b acc
  | op,a,b => .binary p (nativeDType d) op a b

def cast (p : Option ComputeDType) (source target : FloatDType) (x : Expr) : Expr :=
  if nativeDType source = nativeDType target then x
  else .cast p (nativeDType source) (nativeDType target) x

def treeExpr (p : Option ComputeDType) (d : FloatDType) (op : Binary)
    (xs : Fin n → Expr) : Tree n → Expr
  | .input i => xs i
  | .padding w => .leaf (nativeDType d) w
  | .node a b => .binary p (nativeDType d) op (treeExpr p d op xs a) (treeExpr p d op xs b)

def reductionInputs (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → Expr) :
    TileIndex (TileShape.reduceShape shape ax keep) → Fin (TileShape.axisDim shape ax) → Expr :=
  match keep with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

def reduction (plan : Plan) (p : Option ComputeDType) (d : FloatDType) (isMax : Bool)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : TileIndex shape → Expr)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  treeExpr p d (if isMax then .max else .add) (reductionInputs shape ax keep xs i)
    (if isMax then (plan.maxTree ax keep i).tree else (plan.sumTree ax keep i).tree)

/-- SM90 natural-exp lowering is explicit multiply by the actual FP32
log2(e) word followed by the native approximate base-two exponent primitive. -/
def nativeUnary (p : Option ComputeDType) : Unary → Expr → Expr
  | .exp,x => .unary p .exp2 (.binary p .fp32 .mul x (.leaf .fp32 0x3fb8aa3b))
  | op,x => .unary p op x

theorem exp_lowering (p : Option ComputeDType) (x : Expr) :
    nativeUnary p .exp x = .unary p .exp2 (.binary p .fp32 .mul x (.leaf .fp32 0x3fb8aa3b)) := rfl

/-- Every scalar operation is retained as a graph node. Natural-exp lowering
retains its explicit multiply/ex2 nodes, with provenance in the receipt;
no equality to Real.exp is asserted. FP32 load and same-format copy/cast are
word transport, while all actual narrowing casts remain explicit. -/
def algebra (plan : Plan) : Algebra Expr where
  literal := fun p d x => .literal p (nativeDType d) x
  negInf := .leaf .fp32 0xff800000
  binary := consume plan
  unary := nativeUnary
  cast := cast
  fromNat := Expr.fromNat
  fromInt := Expr.fromInt
  fp32Bits := fun w => .leaf .fp32 w.bits
  fp32Load := id
  reduceMax := fun p {_} ax keep xs => reduction plan p .fp32 Bool.true ax keep xs
  reduceSum := fun p {_} ax keep xs => reduction plan p .fp32 Bool.false ax keep xs

def backend (plan : Plan) (eq : Equality) : FPBackend.Ops Expr where
  scalar := algebra plan
  eq := fun p d => eq p (nativeDType d)
  dot := fun p {batch} {m k n} a b i =>
    .pendingDot p .fp32 .fp32 .fp32 (dotOptions .fp32 .fp32 {}) batch m k n a b i
  dotOptions := fun p opts {batch} {m k n} a b i =>
    .pendingDot p .fp32 .fp32 .fp32 (dotOptions .fp32 .fp32 opts) batch m k n a b i
  typedDot := fun p l r out opts {batch} {m k n} a b i =>
    ofFloat out (.pendingDot p (nativeDType l) (nativeDType r) (nativeDType out) (dotOptions l r opts)
      batch m k n (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)
  typedSum := fun p d {_} ax keep xs i =>
    ofFloat d (reduction plan p d Bool.false ax keep (fun j => toFloat d (xs j)) i)
  typedMax := fun p d {_} ax keep xs i =>
    ofFloat d (reduction plan p d Bool.true ax keep (fun j => toFloat d (xs j)) i)

/-- A pending dot is consumed without a separately rounded dot output or add. -/
theorem consume_dot (plan : Plan) (p dp : Option ComputeDType) (d l r out : FloatDType)
    (opts : DotOptions) (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    consume plan p d .add acc (.pendingDot dp l r out opts batch m k n a b i) =
      .dotAcc dp l r out opts batch m k n a b acc i := rfl

/-- The normalizer uses an actual ternary FMA and one separately rounded
product. The first product's standalone orientation is irrelevant because it
is never rounded; the FMA and surviving product retain their own orientations. -/
theorem normalizer (plan : Plan) (p : Option ComputeDType) (d : FloatDType)
    (alpha li beta lij : Expr) :
    consume plan p d .add (consume plan p d .mul alpha li) (consume plan p d .mul beta lij) =
      if plan.reverseFMA p (nativeDType d) alpha li then
        .fma p (nativeDType d) li alpha
          (.product p (nativeDType d) (plan.reverseMul p (nativeDType d) beta lij) beta lij)
      else .fma p (nativeDType d) alpha li
          (.product p (nativeDType d) (plan.reverseMul p (nativeDType d) beta lij) beta lij) := rfl

/-- Complete successful source-grid execution as the staged native graph,
including logical readback, ordered copyback, every original storage cell's
frame and Python None; native integer correspondence is composed publicly.
No floating numerical identity or whole-kernel execution is a premise. -/
theorem whole_call (plan : Plan) (eq : Equality) (c : Host.Call) (storage : Storage) (s : State Expr)
    (logical : FPHost.LogicalInputs Expr) (view : OutputView.View FPHost.Coord Expr)
    (before : Nat → Expr) (prepared : FPHost.Prepared c storage s logical view before)
    (writable : view.Writable (FPHost.active c))
    (buffers : NativeIndex.Buffers) (native : NativeIndex.Ready c storage buffers) :
    (∃ result, FPHost.Exec (backend plan eq) c storage s view before result ∧
      FPHost.Result (backend plan eq) c storage s view before result.2) ∧
      NativeIndex.Correspondence c storage buffers :=
  ⟨FPHost.whole_call (backend plan eq) c storage s logical view before prepared writable,
    NativeIndex.correspondence c storage buffers native⟩

#axiomsClean consume_dot
#axiomsClean normalizer
#axiomsClean whole_call
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph
