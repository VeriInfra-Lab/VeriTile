import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeEvaluate

/-! Consumer-indexed native staging. A logical row can have several physical
replicas with different ordered reductions. The graph retains every replica;
operand routes and final readback select the actual physical consumer. The
two floating `where` guards are delayed graph nodes, so they are evaluated at
that consumer as well. No equality between replicas or reduction trees is used.

The source/native profile interface consists solely of trees, orientations and
discrete owner routes. Its concrete compiler binding is external evidence; the
whole-call theorem does not assert that an arbitrary profile is a compiler.
-/
namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativePhysical
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open NativeGraph (Tree OrderedTree nativeDType dotOptions)
noncomputable section
set_option maxHeartbeats 2000000

/-- A physical replica identifier, interpreted by the selected launch profile.
It is a discrete graph choice and never a floating result. -/
abbrev Owner := Nat

inductive Edge where
  | unary | left | right | factorLeft | factorRight | accumulator
  | guardValue | guardTest | guardReplacement
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
  | guard (precision : Option ComputeDType) (dtype : FloatDType) (value test replacement : Expr)
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

def guardMax (p : Option ComputeDType) (x : Expr) : Expr :=
  .guard p .fp32 x (.literal p .fp32 (-1000000000)) (.literal p .fp32 0)

def guardNormalizer (p : Option ComputeDType) (d : FloatDType) (x : Expr) : Expr :=
  .guard p (nativeDType d) x (.literal p (nativeDType d) 0)
    (.literal p (nativeDType d) (1/1000000000))

def consume (p : Option ComputeDType) (d : FloatDType) : Binary → Expr → Expr → Expr
  | .mul,a,b => .product p (nativeDType d) a b
  | .add,acc,.pendingDot dp l r out opts batch m k n a b i =>
      .dotAcc dp l r out opts batch m k n a b acc i
  | .add,.product _ _ a b,acc => guardNormalizer p d (.fma p (nativeDType d) a b acc)
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
  let node := Expr.reduction p (nativeDType d) (if isMax then .max else .add)
    (TileShape.axisDim shape ax)
    (if isMax then plan.maxTree ax keep i else plan.sumTree ax keep i)
    (reductionInputs shape ax keep xs i)
  if isMax then guardMax p node else node

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

/-- Both source floating equalities are immediate guards already retained by
`guardMax` / `guardNormalizer`. False here is a staging token that transports
the delayed node through the source `where`; it is not a numerical comparison
result. The source audit must check that these are the only floating eq sites.
The actual equality and select are performed by `evaluate` at each owner. -/
def backend (plan : Plan) : FPBackend.Ops Expr where
  scalar := algebra plan
  eq := fun _ _ _ _ => Bool.false
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

def evalTree {α : Type} (binary : α → α → α) (word : BitVec 32 → α)
    (inputs : Fin n → α) : Tree n → α
  | .input i => inputs i
  | .padding w => word w
  | .node a b => binary (evalTree binary word inputs a) (evalTree binary word inputs b)

def evaluate {α : Type} (ops : NativeGraph.Primitives α) (plan : Plan) : Expr → Owner → α
  | .leaf d w,_ => ops.leaf d w
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
  | e@(.guard p d value test replacement),owner =>
      let x := evaluate ops plan value (plan.route e .guardValue owner)
      if ops.eq p d x (evaluate ops plan test (plan.route e .guardTest owner)) then
        evaluate ops plan replacement (plan.route e .guardReplacement owner) else x
  | e@(.reduction p d op n trees inputs),owner =>
      evalTree (ops.binary p d op) (ops.leaf d)
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

theorem delayed_guard (plan : Plan) (p : Option ComputeDType) (d : FloatDType)
    (value test replacement : Expr) :
    (if (backend plan).eq p d (.guard p d value test replacement) test then replacement
      else .guard p d value test replacement) = .guard p d value test replacement := rfl

theorem consume_dot (p dp : Option ComputeDType) (d l r out : FloatDType)
    (opts : DotOptions) (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    consume p d .add acc (.pendingDot dp l r out opts batch m k n a b i) =
      .dotAcc dp l r out opts batch m k n a b acc i := rfl

theorem normalizer (p : Option ComputeDType) (d : FloatDType) (alpha li beta lij : Expr) :
    consume p d .add (consume p d .mul alpha li) (consume p d .mul beta lij) =
      guardNormalizer p d (.fma p (nativeDType d) alpha li (.product p (nativeDType d) beta lij)) := rfl

theorem every_replica_complete (trees : Owner → OrderedTree n) (owner : Owner) :
    (trees owner).tree.indices.Perm (List.finRange n) := (trees owner).complete

def logicalWords (query key value : Nat → Nat → Nat → BitVec 32) : FPHost.LogicalInputs Expr :=
  ⟨fun row head dim => .leaf .fp32 (query row head dim),
   fun row head dim => .leaf .fp32 (key row head dim),
   fun row head dim => .leaf .fp32 (value row head dim)⟩

theorem whole_call {α : Type} (ops : NativeGraph.Primitives α) (plan : Plan)
    (readOwner : FPHost.Coord → Owner) (c : Host.Call) (storage : Storage) (s : State Expr)
    (logical : FPHost.LogicalInputs Expr) (view : OutputView.View FPHost.Coord Expr)
    (before : Nat → Expr) (prepared : FPHost.Prepared c storage s logical view before)
    (writable : view.Writable (FPHost.active c))
    (buffers : NativeIndex.Buffers) (native : NativeIndex.Ready c storage buffers) :
    ∃ result reference,
      FPHost.Exec (backend plan) c storage s view before result ∧
      FPHost.Composition.schedule (backend plan).algebra (FPHost.source c storage) c.grid
        (FPHost.Composition.allPrograms c.grid) s = some reference ∧
      result.1 = none ∧
      (∀ i, FPHost.active c i → evaluate ops plan (view.read result.2 i) (readOwner i) =
        evaluate ops plan (FPHost.values c storage reference i) (readOwner i)) ∧
      (∀ a, (¬∃ i, FPHost.active c i ∧ view.addr i = a) → ∀ owner,
        evaluate ops plan (result.2 a) owner = evaluate ops plan (before a) owner) ∧
      (∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] →
        ∀ off, reference.mem rg off = s.mem rg off) ∧
      NativeIndex.Correspondence c storage buffers := by
  obtain ⟨result,hex,ref,href,hread,hframe,hinputs⟩ :=
    FPHost.whole_call (backend plan) c storage s logical view before prepared writable
  refine ⟨result,ref,hex,href,hex.1,?_,?_,hinputs,NativeIndex.correspondence c storage buffers native⟩
  · intro i hi
    exact congrArg (fun e => evaluate ops plan e (readOwner i)) (hread i hi)
  · intro a ha owner
    exact congrArg (fun e => evaluate ops plan e owner) (hframe a ha)

/-- Readback may select the owner from the stored graph node. In particular a
dot node retains the actual writer's local output coordinate even when host
slices overlap; selecting its owner does not assume equal replica values. -/
def readback {α : Type} (ops : NativeGraph.Primitives α) (plan : Plan)
    (ownerOf : Expr → Owner) (e : Expr) : α := evaluate ops plan e (ownerOf e)

theorem whole_readback {α : Type} (ops : NativeGraph.Primitives α) (plan : Plan)
    (ownerOf : Expr → Owner) (c : Host.Call) (storage : Storage) (s : State Expr)
    (logical : FPHost.LogicalInputs Expr) (view : OutputView.View FPHost.Coord Expr)
    (before : Nat → Expr) (prepared : FPHost.Prepared c storage s logical view before)
    (writable : view.Writable (FPHost.active c))
    (buffers : NativeIndex.Buffers) (native : NativeIndex.Ready c storage buffers) :
    ∃ result reference,
      FPHost.Exec (backend plan) c storage s view before result ∧
      FPHost.Composition.schedule (backend plan).algebra (FPHost.source c storage) c.grid
        (FPHost.Composition.allPrograms c.grid) s = some reference ∧
      result.1 = none ∧
      (∀ i, FPHost.active c i → readback ops plan ownerOf (view.read result.2 i) =
        readback ops plan ownerOf (FPHost.values c storage reference i)) ∧
      (∀ a, (¬∃ i, FPHost.active c i ∧ view.addr i = a) →
        readback ops plan ownerOf (result.2 a) = readback ops plan ownerOf (before a)) ∧
      (∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] →
        ∀ off, reference.mem rg off = s.mem rg off) ∧
      NativeIndex.Correspondence c storage buffers := by
  obtain ⟨result,hex,ref,href,hread,hframe,hinputs⟩ :=
    FPHost.whole_call (backend plan) c storage s logical view before prepared writable
  refine ⟨result,ref,hex,href,hex.1,?_,?_,hinputs,NativeIndex.correspondence c storage buffers native⟩
  · intro i hi
    exact congrArg (readback ops plan ownerOf) (hread i hi)
  · intro a ha
    exact congrArg (readback ops plan ownerOf) (hframe a ha)

#axiomsClean consume_dot
#axiomsClean normalizer
#axiomsClean delayed_guard
#axiomsClean every_replica_complete
#axiomsClean whole_call
#axiomsClean whole_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativePhysical
