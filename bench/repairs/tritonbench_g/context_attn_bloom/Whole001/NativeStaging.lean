import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.NativeInstructions
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.FPCore

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

abbrev Owner := Nat
inductive Edge where
  | left | right | seed | factorLeft | factorRight
  | reduction (i : Nat)
  | dotLeft (i : Nat) | dotRight (i : Nat)
  deriving DecidableEq

/-- Pending source expressions keep a product unrounded until its actual
consumer. In particular the normalizer keeps default fusion and both dot
updates consume the actual preceding accumulator as the first MMA seed. -/
inductive Expr where
  | bits (word : BitVec 32)
  | literal (value : ℝ)
  | binary (op : Binary) (a b : Expr)
  | exponential (x : Expr)
  | materialize (x : Expr)
  | reduction (op : Binary) (n : Nat) (inputs : Fin n → Expr)
  | pendingDot (D row col : Nat) (q k : Fin D → Expr)

instance : Inhabited Expr := ⟨.bits 0⟩

/-- Compiler information only: concrete physical routes, multiplication
operand order and complete ordered reduction trees for each consumer.
No floating function, output word, or execution proposition is a field. -/
structure Plan where
  route : Expr → Edge → Owner → Owner
  reverseMul : Expr → Owner → Bool
  tree : (n : Nat) → 0<n → Expr → Owner → OrderedTree n

def chooseMul (reverse : Bool) (a b : Graph) : Graph :=
  if reverse then .mulRN b a else .mulRN a b

def lower (plan : Plan) : Expr → Owner → Graph
  | .bits b,_ => .bits b
  | .literal x,_ => .literal32 x
  | e@(.binary .add seed (.pendingDot D row col q k)),owner =>
      dot D row col
        (fun i => lower plan (q i) (plan.route e (.dotLeft i.val) owner))
        (fun i => lower plan (k i) (plan.route e (.dotRight i.val) owner))
        (lower plan seed (plan.route e .seed owner))
  | e@(.binary .add (.binary .mul a b) c),owner =>
      let av := lower plan a (plan.route e .factorLeft owner)
      let bv := lower plan b (plan.route e .factorRight owner)
      let cv := lower plan c (plan.route e .seed owner)
      if plan.reverseMul (.binary .mul a b) owner then .fmaRN bv av cv else .fmaRN av bv cv
  | e@(.binary op a b),owner =>
      let av := lower plan a (plan.route e .left owner)
      let bv := lower plan b (plan.route e .right owner)
      match op with
      | .add => .addRN av bv
      | .sub => .subRN av bv
      | .mul => chooseMul (plan.reverseMul e owner) av bv
      | .div => .divFull av bv
      | .max => .maxNum av bv
      | .pow => zero
  | e@(.exponential x),owner =>
      .ex2Approx (.mulRN (lower plan x (plan.route e .left owner)) (.bits 0x3FB8AA3B))
  | e@(.materialize x),owner => lower plan x (plan.route e .left owner)
  | e@(.reduction op n inputs),owner =>
      if h : 0<n then
        (plan.tree n h e owner).tree.fold
          (fun i => lower plan (inputs i) (plan.route e (.reduction i.val) owner))
          (if op=.max then Graph.maxNum else Graph.addRN)
      else zero
  | e@(.pendingDot D row col q k),owner =>
      dot D row col
        (fun i => lower plan (q i) (plan.route e (.dotLeft i.val) owner))
        (fun i => lower plan (k i) (plan.route e (.dotRight i.val) owner)) zero

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i row)

def reduction (op : Binary) (_ : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  .reduction op (TileShape.axisDim shape ax) (reductionInputs ax keep xs i)

def scalar : Algebra Expr where
  literal := fun _ _ x => .literal x
  negInf := .bits 0xFF800000
  binary := fun _ _ op a b => .binary op a b
  unary := fun _ op a => match op with
    | .exp => .exponential a
    | _ => default
  cast := fun _ _ _ a => .materialize a
  fromNat := fun _ n => .literal n
  fromInt := fun _ n => .literal n
  fp32Bits := fun b => .bits b.bits
  fp32Load := fun a => .materialize a
  reduceSum := reduction .add
  reduceMax := reduction .max

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Expr left.toTileDType (batch++[m,k]) →
      Values Expr right.toTileDType (batch++[k,n]) →
      Values Expr out.toTileDType (batch++[m,n])
  | [], _, k, _, a, b, i => ofFloat out (.pendingDot k i.1.val i.2.1.val
      (fun d => toFloat left (a ⟨i.1,d,()⟩)) (fun d => toFloat right (b ⟨d,i.2.1,()⟩)))
  | _::ds, _, _, _, a, b, i => dotValues none left right out {} ds
      (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def dots : FPSuccess.Dot Expr where
  apply := fun p l r out options {batch} {_} {_} {_} a b => dotValues p l r out options batch a b

def backend : Algebra Expr := FPSuccess.withDot scalar dots

theorem every_reduction_leaf (plan : Plan) (n : Nat) (hn : 0<n)
    (e : Expr) (owner : Owner) (i : Fin n) : i ∈ (plan.tree n hn e owner).tree.leaves :=
  reduction_covered _ i

/-- Evaluation of every source-produced native expression follows from local
instruction transport, with the same compiler routes and operand order. -/
theorem lowered_transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (plan : Plan) (e : Expr) (owner : Owner) :
    f ((lower plan e owner).eval A)=(lower plan e owner).eval B := transport f A B h _

#axiomsClean lowered_transport
#axiomsClean every_reduction_leaf
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Native
