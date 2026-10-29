import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeProgramBridge
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPHost

/-! Native floating structure for the default FP32 branch. The vocabulary is
adapted from the Mistral native graph audit, without its extra floating guards.
The source is executed with staged expressions: the normalizer's product is
consumed by an actual FMA before it is rounded. QK has a zero dot accumulator;
PV retains the source's explicit accumulator. Ordered reductions are indexed
by physical consumers. No equation identifies separate rounding with fusion. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
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
structure Plan where
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

/-- Local native ports: FMA is distinct from binary mul/add, and dotAcc retains
the accumulator, formats, TF32 options and exact operand tiles. -/
structure Operations where
  literal : Option ComputeDType → ℝ → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  binary : Option ComputeDType → Binary → Word → Word → Word
  fma : Option ComputeDType → Word → Word → Word → Word
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch ++ [m,k]) → Word) → (TileIndex (batch ++ [k,n]) → Word) →
    Word → TileIndex (batch ++ [m,n]) → Word

def evalTree (binary : Word → Word → Word) (inputs : Fin n → Word) : Tree n → Word
  | .input i => inputs i | .padding w => w
  | .node a b => binary (evalTree binary inputs a) (evalTree binary inputs b)

def evaluate (ops : Operations) (plan : Plan) : Expr → Owner → Word
  | .leaf word,_ => word
  | .zero,_ => 0
  | .negInf,_ => 0xff800000
  | .sentinel,_ => 0xccbebc20
  | .literal p x,_ => ops.literal p x
  | .fromNat p x,_ => ops.fromNat p x
  | .fromInt p x,_ => ops.fromInt p x
  | e@(.cast p source target x),owner => ops.cast p source target (evaluate ops plan x (plan.route e .unary owner))
  | e@(.unary p op x),owner => ops.unary p op (evaluate ops plan x (plan.route e .unary owner))
  | e@(.binary p op x y),owner => ops.binary p op
      (evaluate ops plan x (plan.route e .left owner)) (evaluate ops plan y (plan.route e .right owner))
  | e@(.product p x y),owner =>
      let a := evaluate ops plan x (plan.route e .left owner)
      let b := evaluate ops plan y (plan.route e .right owner)
      if plan.reverseMul p x y owner then ops.binary p .mul b a else ops.binary p .mul a b
  | e@(.fma p x y acc),owner =>
      let a := evaluate ops plan x (plan.route e .factorLeft owner)
      let b := evaluate ops plan y (plan.route e .factorRight owner)
      let v := evaluate ops plan acc (plan.route e .accumulator owner)
      if plan.reverseFMA p x y owner then ops.fma p b a v else ops.fma p a b v
  | e@(.reduction p op n trees inputs),owner =>
      evalTree (ops.binary p op)
        (fun i => evaluate ops plan (inputs i) (plan.route e (.reductionInput i.val) owner)) (trees owner).tree
  | e@(.dotAcc p left right out options batch m k n a b acc i),owner =>
      ops.dotAcc p left right out options batch m k n
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (evaluate ops plan acc (plan.route e .accumulator owner)) i

theorem normalizer (p : Option ComputeDType) (li alpha lij : Expr) :
    consume p .add (consume p .mul li alpha) lij = .fma p li alpha lij := rfl

theorem each_owner_tree_complete (trees : Owner → OrderedTree n) (owner : Owner) :
    (trees owner).tree.indices.Perm (List.finRange n) := (trees owner).complete

/-- Native graph execution is the complete logical source invocation under
staged instruction semantics. It is not equality to separately rounded source
arithmetic. The integer bridge is instantiated with this actual graph backend. -/
theorem successful_grid (plan : Plan) (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (valid : Host.Valid c A logical) (w : Native.Widths) (B : Native.Bytes A) (abi : Native.ABI c A w B)
    (s : State Expr) (metadata : Native.FullMetadata c s) :
    Native.GridBridge c A logical w B (backend plan) s ∧
    ∃ t, FPSuccess.runGrid c (backend plan) (FPSuccess.allPrograms c) s = some t ∧
      (∀ rg n, ¬ FPSuccess.writes c (FPSuccess.allPrograms c) rg n → t.mem rg n = s.mem rg n) := by
  refine ⟨Native.grid_bridge c A logical valid w B abi (backend plan) (dots.support (scalar plan) c.layout) s metadata,?_⟩
  obtain ⟨t,ht,hm,hf⟩ := FPSuccess.grid_success_frame c (backend plan)
    (dots.support (scalar plan) c.layout) valid.input_separate (FPSuccess.allPrograms c) s metadata.1
  exact ⟨t,ht,hf⟩

def scratch (ops : Operations) (plan : Plan) (c : Host.Call)
    (owner : FPHost.Coordinate c → Owner) (t : State Expr) (i : FPHost.Coordinate c) : Word :=
  evaluate ops plan ((t.mem c.regions.out (c.output i.1 i.2.1 i.2.2.1 i.2.2.2)).read .real) (owner i)

def result (ops : Operations) (plan : Plan) (c : Host.Call)
    (owner : FPHost.Coordinate c → Owner) (o : FPViews.Output (FPHost.Coordinate c)) (t : State Expr) :=
  ((none : Option Unit), o.owner, (FPHost.capture c o).copyBack (scratch ops plan c owner t))

theorem evaluated_call (ops : Operations) (plan : Plan) (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (valid : Host.Valid c A logical) (w : Native.Widths) (B : Native.Bytes A) (abi : Native.ABI c A w B)
    (s : State Expr) (metadata : Native.FullMetadata c s) (owner : FPHost.Coordinate c → Owner)
    (o : FPViews.Output (FPHost.Coordinate c))
    (unique : ∀ i j, (FPHost.capture c o).active i → (FPHost.capture c o).active j → o.address i = o.address j → i = j) :
    Native.GridBridge c A logical w B (backend plan) s ∧
    ∃ t, FPSuccess.runGrid c (backend plan) (FPSuccess.allPrograms c) s = some t ∧
      (∀ i, (FPHost.capture c o).active i →
        (FPHost.capture c o).logical (result ops plan c owner o t).2.2 i = scratch ops plan c owner t i) ∧
      (∀ cell, (¬∃ i, (FPHost.capture c o).active i ∧ o.address i = cell) →
        (result ops plan c owner o t).2.2 cell = o.initial cell) ∧
      (result ops plan c owner o t).1 = none ∧ (result ops plan c owner o t).2.1 = o.owner ∧
      (∀ rg n, ¬ FPSuccess.writes c (FPSuccess.allPrograms c) rg n → t.mem rg n = s.mem rg n) := by
  obtain ⟨bridge,t,ht,hframe⟩ := successful_grid plan c A logical valid w B abi s metadata
  exact ⟨bridge,t,ht,(FPHost.capture c o).copyBack_read _ unique,
    (FPHost.capture c o).copyBack_frame _,rfl,rfl,hframe⟩

#print axioms evaluated_call
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.NativeGraph
