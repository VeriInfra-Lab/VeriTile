import bench.tritonbench_g.context_attn_fwd.Proofs.FPGrid
import bench.tritonbench_g.context_attn_fwd.Proofs.FPTotalBackend

/-! Native floating structure for the default FP32 branch. The vocabulary is
adapted from the Mistral native graph audit, without its extra floating guards.
The source is executed with staged expressions: the normalizer's product is
consumed by an actual FMA before it is rounded. QK has a zero dot accumulator;
PV retains the source's explicit accumulator. Ordered reductions are indexed
by physical consumers. No equation identifies separate rounding with fusion. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

abbrev Word := BitVec 32
abbrev Owner := Nat

inductive Tree (n : Nat) where
  | input (i : Fin n)
  | padding (word : Word)
  | node (a b : Tree n)

def Tree.indices : Tree n → List (Fin n)
  | .input i => [i] | .padding _ => [] | .node a b => a.indices ++ b.indices

structure OrderedTree (n : Nat) where
  tree : Tree n
  complete : tree.indices.Perm (List.finRange n)

inductive Edge where
  | unary | left | right | factorLeft | factorRight | accumulator
  | reductionInput (column : Nat)
  | dotLeft (linear : Nat) | dotRight (linear : Nat)
  deriving DecidableEq

inductive Expr where
  | leaf (word : Word)
  | zero | negInf | sentinel
  | literal (precision : Option ComputeDType) (value : ℝ)
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | fromInt (precision : Option ComputeDType) (value : Int)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (x : Expr)
  | binary (precision : Option ComputeDType) (op : Binary) (x y : Expr)
  | product (precision : Option ComputeDType) (x y : Expr)
  | fma (precision : Option ComputeDType) (x y acc : Expr)
  | reduction (precision : Option ComputeDType) (op : Binary)
      (n : Nat) (trees : Owner → OrderedTree n) (inputs : Fin n → Expr)
  | dotAcc (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch ++ [m,k]) → Expr) (b : TileIndex (batch ++ [k,n]) → Expr)
      (acc : Expr) (i : TileIndex (batch ++ [m,n]))

instance : Inhabited Expr := ⟨.zero⟩

/-- Only structural data: physical routes, ordered trees and operand order.
No floating value, output, or execution callback occurs in this profile. -/
inductive MMAWidth where
  | n16 | n32 | n64 | n128 | n256
  deriving DecidableEq

def MMAWidth.columns : MMAWidth → Nat
  | .n16 => 16 | .n32 => 32 | .n64 => 64 | .n128 => 128 | .n256 => 256

theorem MMAWidth.positive (w : MMAWidth) : 0<w.columns := by cases w <;> decide

structure Plan where
  matrixWidth : Nat → MMAWidth
  reverseMul : Option ComputeDType → Expr → Expr → Owner → Bool
  reverseFMA : Option ComputeDType → Expr → Expr → Owner → Bool
  sumTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Owner → OrderedTree (TileShape.axisDim shape ax)
  maxTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Owner → OrderedTree (TileShape.axisDim shape ax)
  route : Expr → Edge → Owner → Owner

def nativeDType : FloatDType → FloatDType | .real => .fp32 | d => d
def dotOptions (source : DotOptions) : DotOptions :=
  if source.inputPrecision = none then {source with inputPrecision := some "tf32"} else source

def literal (p : Option ComputeDType) (x : ℝ) : Expr := by
  classical
  exact if x = 0 then .zero else if x = -100000000 then .sentinel else .literal p x

/-- The initial +0 plus -infinity is the compiler's literal fold. Only the
normalizer's left-product/add pattern contracts; surviving products remain. -/
def consume (p : Option ComputeDType) : Binary → Expr → Expr → Expr
  | .mul,a,b => .product p a b
  | .add,.zero,.negInf => .negInf
  | .add,.product _ a b,acc => .fma p a b acc
  | op,a,b => .binary p op a b

def cast (p : Option ComputeDType) (source target : FloatDType) (x : Expr) : Expr :=
  if nativeDType source = nativeDType target then x
  else .cast p (nativeDType source) (nativeDType target) x

def reductionInputs (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → Expr) :
    TileIndex (TileShape.reduceShape shape ax keep) → Fin (TileShape.axisDim shape ax) → Expr :=
  match keep with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

def reduction (plan : Plan) (p : Option ComputeDType) (isMax : Bool)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : TileIndex shape → Expr)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  .reduction p (if isMax then .max else .add) (TileShape.axisDim shape ax)
    (if isMax then plan.maxTree ax keep i else plan.sumTree ax keep i)
    (reductionInputs shape ax keep xs i)

def scalar (plan : Plan) : Algebra Expr where
  literal := fun p _ x => literal p x
  negInf := .negInf
  binary := fun p _ => consume p
  unary := Expr.unary
  cast := cast
  fromNat := Expr.fromNat
  fromInt := Expr.fromInt
  fp32Bits := fun w => .leaf w.bits
  fp32Load := id
  reduceMax := fun p {_} ax keep xs => reduction plan p Bool.true ax keep xs
  reduceSum := fun p {_} ax keep xs => reduction plan p Bool.false ax keep xs

def dots : FPSuccess.TotalDots Expr where
  dot := fun p {batch} {m k n} a b i =>
    .dotAcc p .fp32 .fp32 .fp32 (dotOptions {}) batch m k n a b .zero i
  dotAcc := fun p left right out options {batch} {m k n} a b acc i =>
    ofFloat out (.dotAcc p (nativeDType left) (nativeDType right) (nativeDType out) (dotOptions options)
      batch m k n (fun j => toFloat left (a j)) (fun j => toFloat right (b j)) (toFloat out (acc i)) i)

def backend (plan : Plan) : Algebra Expr := dots.install (scalar plan)

/-- Only scalar FP32 instructions and one output coordinate of one fixed
SM90 WGMMA m64n{16,32,64,128,256}k8 TF32 instruction are numerical ports.
An arbitrary-size source dot is the explicit K/8 instruction chain below. -/
structure ScalarOps where
  literal32 : ℝ → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  fmaRN : Word → Word → Word → Word
  wgmmaTF32 : (width : MMAWidth) → Fin 64 → Fin width.columns →
    (Fin 8 → Word) → (Fin 8 → Word) → Bool → Word → Word

def binary (ops : ScalarOps) : Binary → Word → Word → Word
  | .add => ops.add | .sub => ops.sub | .mul => ops.mul
  | .div => ops.div | .max => ops.max | _ => fun _ _ => 0

def dotWord (ops : ScalarOps) (width : MMAWidth) (row column k : Nat)
    (a b : Fin k → Word) (accumulate : Bool) (seed : Word) : Word :=
  (List.range ((k+7)/8)).foldl (fun previous block =>
    ops.wgmmaTF32 width ⟨row%64,Nat.mod_lt _ (by decide)⟩
      ⟨column%width.columns,Nat.mod_lt _ width.positive⟩
      (fun lane => if h : block*8+lane.val<k then a ⟨block*8+lane.val,h⟩ else 0)
      (fun lane => if h : block*8+lane.val<k then b ⟨block*8+lane.val,h⟩ else 0)
      (if block=0 then accumulate else Bool.true) previous) seed

/-- Batched shape recursion does not combine floating results. Each output
has its actual fixed-instruction K-chain and explicit zero/live accumulator. -/
def dotAt (ops : ScalarOps) (plan : Plan) : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    Bool → Word → TileIndex (batch++[m,n]) → Word
  | [],_,k,n,a,b,accumulate,seed,i => dotWord ops (plan.matrixWidth n) i.1.val i.2.1.val k
      (fun j => a ⟨i.1,j,()⟩) (fun j => b ⟨j,i.2.1,()⟩) accumulate seed
  | _::ds,_,_,_,a,b,accumulate,seed,i => dotAt ops plan ds (fun j => a ⟨i.1,j⟩)
      (fun j => b ⟨i.1,j⟩) accumulate seed i.2

/-- WGMMA scale-d is determined by the source accumulator constructor,
never by the evaluated word. A live accumulator remains live at +0. -/
def accumulate : Expr → Bool
  | .zero => Bool.false
  | _ => Bool.true

theorem feature_coverage (k d : Nat) (hd : d<k) :
    d/8 ∈ List.range ((k+7)/8) ∧ d%8<8 ∧ d/8*8+d%8=d := by
  refine ⟨List.mem_range.mpr ?_,Nat.mod_lt _ (by decide),?_⟩
  · omega
  · simpa [Nat.mul_comm] using Nat.div_add_mod d 8

theorem feature_unique (d block lane : Nat) (hl : lane<8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 := by omega

def evalTree (binary : Word → Word → Word) (inputs : Fin n → Word) : Tree n → Word
  | .input i => inputs i | .padding w => w
  | .node a b => binary (evalTree binary inputs a) (evalTree binary inputs b)

def evaluate (ops : ScalarOps) (plan : Plan) : Expr → Owner → Word
  | .leaf word,_ => word
  | .zero,_ => 0
  | .negInf,_ => 0xff800000
  | .sentinel,_ => 0xccbebc20
  | .literal _ x,_ => ops.literal32 x
  | .fromNat _ _,_ => 0
  | .fromInt _ _,_ => 0
  | e@(.cast _ source target x),owner =>
      if source=target then evaluate ops plan x (plan.route e .unary owner) else 0
  | e@(.unary _ op x),owner =>
      if op=.exp2 then ops.exp2 (evaluate ops plan x (plan.route e .unary owner)) else 0
  | e@(.binary _ op x y),owner => binary ops op
      (evaluate ops plan x (plan.route e .left owner)) (evaluate ops plan y (plan.route e .right owner))
  | e@(.product p x y),owner =>
      let a := evaluate ops plan x (plan.route e .left owner)
      let b := evaluate ops plan y (plan.route e .right owner)
      if plan.reverseMul p x y owner then ops.mul b a else ops.mul a b
  | e@(.fma p x y acc),owner =>
      let a := evaluate ops plan x (plan.route e .factorLeft owner)
      let b := evaluate ops plan y (plan.route e .factorRight owner)
      let v := evaluate ops plan acc (plan.route e .accumulator owner)
      if plan.reverseFMA p x y owner then ops.fmaRN b a v else ops.fmaRN a b v
  | e@(.reduction p op n trees inputs),owner =>
      evalTree (binary ops op)
        (fun i => evaluate ops plan (inputs i) (plan.route e (.reductionInput i.val) owner)) (trees owner).tree
  | e@(.dotAcc p left right out options batch m k n a b acc i),owner =>
      dotAt ops plan batch
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (accumulate acc)
        (evaluate ops plan acc (plan.route e .accumulator owner)) i

theorem normalizer (p : Option ComputeDType) (li alpha lij : Expr) :
    consume p .add (consume p .mul li alpha) lij = .fma p li alpha lij := rfl

theorem each_owner_tree_complete (trees : Owner → OrderedTree n) (owner : Owner) :
    (trees owner).tree.indices.Perm (List.finRange n) := (trees owner).complete


end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph
