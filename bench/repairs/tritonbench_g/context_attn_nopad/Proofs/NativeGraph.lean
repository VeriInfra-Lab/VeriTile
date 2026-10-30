import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeGraph
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPHostPrecise

/-! Staged native floating graph, reusing the checked Mistral graph vocabulary.
The inherited expression constructors retain default TF32 dot-accumulators,
FMA, the surviving rounded product, explicit exp2/log2(e) lowering and ordered
reduction trees. Staging does not assert any equality between two rounded
operations and a fused operation. Native profile/provenance is separate.

The entry binds every Q/K/V coordinate to an initial logical FP32 word, exact
integer metadata, and initialized output scratch. The source execution and
complete caller/flat-memory frame are reused with these graph payloads.
-/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
namespace Graph
export VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph
  (Expr Plan Tree OrderedTree consume algebra dotOptions nativeUnary)
end Graph
noncomputable section

abbrev Word := BitVec 32
abbrev Expr := Graph.Expr
abbrev Plan := Graph.Plan

def backend (plan : Plan) : FPBackend.Ops Expr where
  scalar := Graph.algebra plan
  dot := fun p {batch} {m k n} a b i =>
    .pendingDot p .fp32 .fp32 .fp32 (Graph.dotOptions .fp32 .fp32 {}) batch m k n a b i

/-- Initial logical values, before any raw pointer is passed to the kernel. -/
structure Inputs where
  query : Nat → Nat → Nat → Word
  key : Nat → Nat → Nat → Word
  value : Nat → Nat → Nat → Word

/-- Ordinary successful coordinate preparation; no result or execution law.
Logical negative input views have already supplied their resolved FP32 words.
The caller output involution is the existing ordinary tensor-view contract. -/
structure Prepared (c : Call) (r : Regions) (s : State Expr) (input : Inputs)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr) : Prop where
  metadata : FPHostSource.Metadata c r s
  query : ∀ row head dim, row < c.q.rows → head < c.q.heads → dim < c.q.dims →
    (s.mem r.query (c.q.offset row head dim)).read .real = .leaf .fp32 (input.query row head dim)
  key : ∀ row head dim, row < c.k.rows → head < c.k.heads → dim < c.k.dims →
    (s.mem r.key (c.k.offset row head dim)).read .real = .leaf .fp32 (input.key row head dim)
  value : ∀ row head dim, row < c.v.rows → head < c.v.heads → dim < c.v.dims →
    (s.mem r.value (c.v.offset row head dim)).read .real = .leaf .fp32 (input.value row head dim)
  initialized : FPHostPrecise.Initialized c r s view before

/-- Both source dot/add sites construct one native dot-accumulator node.
The dot callback returns a pending graph, not a separately rounded FP result. -/
theorem consumes_dot (plan : Plan) (p : Option ComputeDType)
    (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch ++ [m,k]) → Expr) (b : TileIndex (batch ++ [k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch ++ [m,n])) :
    (backend plan).algebra.binary p .real .add acc ((backend plan).dot p a b i) =
      .dotAcc p .fp32 .fp32 .fp32 (Graph.dotOptions .fp32 .fp32 {}) batch m k n a b acc i := rfl

/-- The source normalizer builds one native FMA and the other rounded product;
its operand orientations are explicit structural native-profile data. -/
theorem normalizer (plan : Plan) (p : Option ComputeDType) (alpha li beta lij : Expr) :
    (backend plan).algebra.binary p .real .add
      ((backend plan).algebra.binary p .real .mul alpha li)
      ((backend plan).algebra.binary p .real .mul beta lij) =
    if plan.reverseFMA p .fp32 alpha li then
      .fma p .fp32 li alpha (.product p .fp32 (plan.reverseMul p .fp32 beta lij) beta lij)
    else .fma p .fp32 alpha li (.product p .fp32 (plan.reverseMul p .fp32 beta lij) beta lij) := rfl

/-- A total local interpretation of graph primitives. Dot-accumulator and FMA
are independent native operations; there are no decomposition axioms. -/
structure Operations where
  literal : Option ComputeDType → FloatDType → ℝ → Word
  fromNat : Option ComputeDType → Nat → Word
  fromInt : Option ComputeDType → Int → Word
  cast : Option ComputeDType → FloatDType → FloatDType → Word → Word
  unary : Option ComputeDType → Unary → Word → Word
  binary : Option ComputeDType → FloatDType → Binary → Word → Word → Word
  fma : Option ComputeDType → FloatDType → Word → Word → Word → Word
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch ++ [m,k]) → Word) → (TileIndex (batch ++ [k,n]) → Word) →
    Word → TileIndex (batch ++ [m,n]) → Word

def eval (ops : Operations) : Expr → Word
  | .leaf _ word => word
  | .literal p d x => ops.literal p d x
  | .fromNat p n => ops.fromNat p n
  | .fromInt p n => ops.fromInt p n
  | .cast p a b x => ops.cast p a b (eval ops x)
  | .unary p op x => ops.unary p op (eval ops x)
  | .binary p d op a b => ops.binary p d op (eval ops a) (eval ops b)
  | .product p d swapped a b =>
      if swapped then ops.binary p d .mul (eval ops b) (eval ops a)
      else ops.binary p d .mul (eval ops a) (eval ops b)
  | .fma p d a b acc => ops.fma p d (eval ops a) (eval ops b) (eval ops acc)
  | .pendingDot p l r out opts batch m k n a b i =>
      ops.dotAcc p l r out opts batch m k n (fun j => eval ops (a j)) (fun j => eval ops (b j))
        (ops.literal p out 0) i
  | .dotAcc p l r out opts batch m k n a b acc i =>
      ops.dotAcc p l r out opts batch m k n (fun j => eval ops (a j)) (fun j => eval ops (b j))
        (eval ops acc) i

@[simp] theorem eval_leaf (ops : Operations) (word : Word) :
    eval ops (.leaf .fp32 word) = word := rfl

theorem prepared_query (ops : Operations) {c : Call} {r : Regions} {s : State Expr}
    {input : Inputs} {view : FPHostSource.OutputView Expr c} {before : Nat → Expr}
    (h : Prepared c r s input view before) (row head dim : Nat)
    (hr : row < c.q.rows) (hh : head < c.q.heads) (hd : dim < c.q.dims) :
    eval ops ((s.mem r.query (c.q.offset row head dim)).read .real) = input.query row head dim := by
  rw [h.query row head dim hr hh hd,eval_leaf]

/-- Complete staged execution from the bound logical input words, with exact
integer/grid correspondence and the full caller/flat-memory frame. This uses
the prepared int32 domain; the wider entries are WideGraph and NativePhysical. -/
theorem whole_call (plan : Plan) {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {s : State Expr} (input : Inputs)
    (hm : Host.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr)
    (prepared : Prepared c r s input view before) :
    (∀ j : HostPlan.Job (FPHostSource.effective c r s),
      NativeProgram.Projects (FPHostSource.effective c r s) r A layoutState j) ∧
    FPHostPrecise.Contract (backend plan).algebra c r A layoutState s view before :=
  ⟨FPHostSource.finite_source_projection hm ha prepared.metadata,
    FPHostPrecise.whole_host_precise (backend plan) hm ha prepared.metadata view before prepared.initialized⟩

/-- Evaluating the already successful graph preserves exact caller readback
and every framed cell for arbitrary local native primitive implementations.
No separate-rounded FP interpreter is identified with this graph evaluator. -/
theorem evaluated_call (plan : Plan) (ops : Operations) {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {s : State Expr} (input : Inputs)
    (hm : Host.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (view : FPHostSource.OutputView Expr c) (before : Nat → Expr)
    (prepared : Prepared c r s input view before) :
    ∃ (raw : State Expr) (result : FPHostSource.Result Expr),
      FPHostSource.runHost (backend plan).algebra c r s view before = some result ∧
      FPHostSource.reference (backend plan).algebra c r s view before = some result ∧
      result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧ result.pythonReturn = none ∧
      (∀ i, eval ops (view.action (result.memory (view.address i))) =
        eval ops ((raw.mem r.out (Logical.coordOffset c i)).read .real)) ∧
      (∀ addr, (¬ ∃ i, Logical.Written c i ∧ view.address i = addr) →
        eval ops (result.memory addr) = eval ops (before addr)) ∧
      (∀ rg off, ¬ (FPHostSource.effective c r s).writes r A layoutState (A.flat,A.addr rg off) →
        raw.mem rg off = s.mem rg off) := by
  obtain ⟨raw,result,_,hr,href,_,hid,halloc,hret,hread,hframe,hflat⟩ :=
    (whole_call plan input hm ha view before prepared).2
  refine ⟨raw,result,hr,href,hid,halloc,hret,?_,?_,hflat⟩
  · intro i
    rw [hread i]
  · intro addr h
    rw [hframe addr h]

#print axioms whole_call
#print axioms evaluated_call
end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeGraph
