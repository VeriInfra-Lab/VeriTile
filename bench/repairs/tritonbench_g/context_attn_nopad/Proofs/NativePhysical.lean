import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideGraph

/-! Physical-owner graph adapted from frozen Mistral NativePhysical snapshot002.
Nopad has no floating where guards. Reduction replicas, operand routes, seeded
default-TF32 dots, FMA and explicit exp2 lowering remain separate graph nodes.
Profile selection is discrete structural evidence, not a numerical axiom. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativePhysical
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph (Tree OrderedTree nativeDType dotOptions)
noncomputable section
set_option maxHeartbeats 2000000

abbrev Owner := Nat

inductive Edge where
  | unary | left | right | factorLeft | factorRight | accumulator
  | reductionInput (column : Nat)
  | dotLeft (linear : Nat) | dotRight (linear : Nat)
  deriving DecidableEq

inductive Expr where
  | leaf (dtype : FloatDType) (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | fromInt (precision : Option ComputeDType) (value : Int)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | product (precision : Option ComputeDType) (dtype : FloatDType) (x y : Expr)
  | fma (precision : Option ComputeDType) (dtype : FloatDType) (x y acc : Expr)
  | reduction (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
      (n : Nat) (trees : Owner → OrderedTree n) (inputs : Fin n → Expr)
  | pendingDot (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (precision : Option ComputeDType) (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (acc : Expr) (i : TileIndex (batch++[m,n]))

instance : Inhabited Expr := ⟨.leaf .fp32 0⟩

structure Plan where
  reverseMul : Option ComputeDType → FloatDType → Expr → Expr → Owner → Bool
  reverseFMA : Option ComputeDType → FloatDType → Expr → Expr → Owner → Bool
  sumTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Owner → OrderedTree (TileShape.axisDim shape ax)
  maxTree : {shape : TileShape} → (ax : Fin shape.length) → (keep : Bool) →
    TileIndex (TileShape.reduceShape shape ax keep) → Owner → OrderedTree (TileShape.axisDim shape ax)
  route : Expr → Edge → Owner → Owner

def consume (p : Option ComputeDType) (d : FloatDType) : Binary → Expr → Expr → Expr
  | .mul,a,b => .product p (nativeDType d) a b
  | .add,acc,.pendingDot dp l r out opts batch m k n a b i =>
      .dotAcc dp l r out opts batch m k n a b acc i
  | .add,.product _ _ a b,acc => .fma p (nativeDType d) a b acc
  | op,a,b => .binary p (nativeDType d) op a b

def cast (p : Option ComputeDType) (source target : FloatDType) (x : Expr) : Expr :=
  if nativeDType source = nativeDType target then x
  else .cast p (nativeDType source) (nativeDType target) x

def reductionInputs (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → Expr) :
    TileIndex (TileShape.reduceShape shape ax keep) → Fin (TileShape.axisDim shape ax) → Expr :=
  match keep with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

def reduction (plan : Plan) (p : Option ComputeDType) (d : FloatDType) (isMax : Bool)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : TileIndex shape → Expr)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  Expr.reduction p (nativeDType d) (if isMax then .max else .add)
    (TileShape.axisDim shape ax)
    (if isMax then plan.maxTree ax keep i else plan.sumTree ax keep i)
    (reductionInputs shape ax keep xs i)

def nativeUnary (p : Option ComputeDType) : Unary → Expr → Expr
  | .exp,x => .unary p .exp2 (.binary p .fp32 .mul x (.leaf .fp32 0x3fb8aa3b))
  | op,x => .unary p op x

def algebra (plan : Plan) : Algebra Expr where
  literal := fun p d x => .literal p (nativeDType d) x
  negInf := .leaf .fp32 0xff800000
  binary := consume
  unary := nativeUnary
  cast := cast
  fromNat := Expr.fromNat
  fromInt := Expr.fromInt
  fp32Bits := fun w => .leaf .fp32 w.bits
  fp32Load := id
  reduceMax := fun p {_} ax keep xs => reduction plan p .fp32 Bool.true ax keep xs
  reduceSum := fun p {_} ax keep xs => reduction plan p .fp32 Bool.false ax keep xs

def backend (plan : Plan) : FPBackend.Ops Expr where
  scalar := algebra plan
  dot := fun p {batch} {m k n} a b i =>
    .pendingDot p .fp32 .fp32 .fp32 (dotOptions .fp32 .fp32 {}) batch m k n a b i

def evalTree {α : Type} (binary : α → α → α) (word : BitVec 32 → α)
    (inputs : Fin n → α) : Tree n → α
  | .input i => inputs i
  | .padding w => word w
  | .node a b => binary (evalTree binary word inputs a) (evalTree binary word inputs b)

def evaluate (ops : NativeGraph.Operations) (plan : Plan) : Expr → Owner → BitVec 32
  | .leaf _ w,_ => w
  | .literal p d x,_ => ops.literal p d x
  | .fromNat p x,_ => ops.fromNat p x
  | .fromInt p x,_ => ops.fromInt p x
  | e@(.cast p source target x),owner =>
      ops.cast p source target (evaluate ops plan x (plan.route e .unary owner))
  | e@(.unary p op x),owner => ops.unary p op (evaluate ops plan x (plan.route e .unary owner))
  | e@(.binary p d op x y),owner => ops.binary p d op
      (evaluate ops plan x (plan.route e .left owner)) (evaluate ops plan y (plan.route e .right owner))
  | e@(.product p d x y),owner =>
      let a := evaluate ops plan x (plan.route e .left owner)
      let b := evaluate ops plan y (plan.route e .right owner)
      if plan.reverseMul p d x y owner then ops.binary p d .mul b a else ops.binary p d .mul a b
  | e@(.fma p d x y acc),owner =>
      let a := evaluate ops plan x (plan.route e .factorLeft owner)
      let b := evaluate ops plan y (plan.route e .factorRight owner)
      let c := evaluate ops plan acc (plan.route e .accumulator owner)
      if plan.reverseFMA p d x y owner then ops.fma p d b a c else ops.fma p d a b c
  | e@(.reduction p d op n trees inputs),owner =>
      evalTree (ops.binary p d op) id
        (fun i => evaluate ops plan (inputs i) (plan.route e (.reductionInput i.val) owner)) (trees owner).tree
  | e@(.pendingDot p l r out opts batch m k n a b i),owner =>
      ops.dotAcc p l r out opts batch m k n
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (ops.literal p out 0) i
  | e@(.dotAcc p l r out opts batch m k n a b acc i),owner =>
      ops.dotAcc p l r out opts batch m k n
        (fun j => evaluate ops plan (a j) (plan.route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops plan (b j) (plan.route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (evaluate ops plan acc (plan.route e .accumulator owner)) i


structure Prepared (c : Call) (r : Regions) (s : State Expr) (input : NativeGraph.Inputs)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr) : Prop where
  metadata : FPHostSource.Metadata c r s
  query : ∀ row head dim, row < c.q.rows → head < c.q.heads → dim < c.q.dims →
    (s.mem r.query (c.q.offset row head dim)).read .real = .leaf .fp32 (input.query row head dim)
  key : ∀ row head dim, row < c.k.rows → head < c.k.heads → dim < c.k.dims →
    (s.mem r.key (c.k.offset row head dim)).read .real = .leaf .fp32 (input.key row head dim)
  value : ∀ row head dim, row < c.v.rows → head < c.v.heads → dim < c.v.dims →
    (s.mem r.value (c.v.offset row head dim)).read .real = .leaf .fp32 (input.value row head dim)
  initialized : FPHostPrecise.Initialized c r s view before

theorem consumes_dot (plan : Plan) (p : Option ComputeDType)
    (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    (backend plan).algebra.binary p .real .add acc ((backend plan).dot p a b i) =
      .dotAcc p .fp32 .fp32 .fp32 (dotOptions .fp32 .fp32 {}) batch m k n a b acc i := rfl

theorem normalizer (plan : Plan) (p : Option ComputeDType) (alpha li beta lij : Expr) :
    (backend plan).algebra.binary p .real .add
      ((backend plan).algebra.binary p .real .mul alpha li)
      ((backend plan).algebra.binary p .real .mul beta lij) =
      .fma p .fp32 alpha li (.product p .fp32 beta lij) := rfl

theorem every_replica_complete (trees : Owner → OrderedTree n) (owner : Owner) :
    (trees owner).tree.indices.Perm (List.finRange n) := (trees owner).complete

def readback (ops : NativeGraph.Operations) (plan : Plan) (ownerOf : Expr → Owner) (e : Expr) : BitVec 32 :=
  evaluate ops plan e (ownerOf e)

/-- Bound logical FP32 leaves, complete successful staged execution, actual
stored graph readback and exact caller/flat frames over the int32/int64 domain.
Each stored node selects its physical owner; no replica equality is assumed. -/
theorem whole_call (plan : Plan) (ops : NativeGraph.Operations) (ownerOf : Expr → Owner)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State Expr}
    (input : NativeGraph.Inputs) (hm : WideDomain.Memory c r A layoutState)
    (ha : NativeProgram.ABI c A) (types : WideNative.MetadataTypes)
    (fits : WideNative.MetadataFits c types) (alloc : WideNative.MetadataAlloc r A types)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr)
    (prepared : Prepared c r s input view before) :
    FPHostPrecise.Contract (backend plan).algebra c r A layoutState s view before ∧
    (∀ j : HostPlan.Job (FPHostSource.effective c r s),
      WideNative.Projects (FPHostSource.effective c r s) r A layoutState types j ∧
      ∀ iter, iter ≤ (HostPlan.global j.2.2).val →
        (NativeProgram.word (iter*128) + NativeProgram.word 128).toInt = ((iter*128+128 : Nat) : Int)) ∧
    (∀ b, b < c.batches →
      (NativeProgram.word b * NativeProgram.word types.start.bytes).toInt = ((b*types.start.bytes : Nat) : Int) ∧
      (NativeProgram.word b * NativeProgram.word types.length.bytes).toInt = ((b*types.length.bytes : Nat) : Int)) ∧
    ∃ (raw : State Expr) (result : FPHostSource.Result Expr),
      FPHostSource.runHost (backend plan).algebra c r s view before = some result ∧
      (∀ i, readback ops plan ownerOf (view.action (result.memory (view.address i))) =
        readback ops plan ownerOf ((raw.mem r.out (Logical.coordOffset c i)).read .real)) ∧
      (∀ addr, (¬ ∃ i, Logical.Written c i ∧ view.address i = addr) →
        readback ops plan ownerOf (result.memory addr) = readback ops plan ownerOf (before addr)) := by
  have hc := WideFP.whole_host_precise (backend plan) hm ha fits prepared.metadata view before prepared.initialized
  refine ⟨hc,?_,WideNative.metadata_byte_offsets hm types alloc,?_⟩
  · have he : FPHostSource.effective c r s = HostPlan.effective c r layoutState := by
      unfold FPHostSource.effective HostPlan.effective
      rw [FPHostSource.metadataMaximum_eq prepared.metadata,WideDomain.metadataMaximum_eq hm]
    rw [he]
    exact WideNative.complete_projection hm ha types fits
  · obtain ⟨raw,result,_,hr,_,_,_,_,_,hread,hframe,_⟩ := hc
    exact ⟨raw,result,hr,fun i => congrArg (readback ops plan ownerOf) (hread i),
      fun addr h => congrArg (readback ops plan ownerOf) (hframe addr h)⟩

#print axioms consumes_dot
#print axioms normalizer
#print axioms whole_call
end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativePhysical
