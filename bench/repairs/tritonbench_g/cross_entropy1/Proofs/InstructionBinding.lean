import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeSelectedReturn

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- One selected instruction's opcode and exact input register identities.
No constructor contains a computed arithmetic result or an expected output. -/
inductive Opcode (σ : Type) where
  | raw (bits : Word)
  | load (state : σ) (tensor : Tensor) (index : Nat)
  | literal (value : ℝ)
  | natural (value : Nat)
  | negativeInfinity
  | binary (op : ScalarBinary) (left right : Nat)
  | unary (op : ScalarUnary) (value : Nat)
  | fma (left right addend : Nat)
  | hostScale (scale : ℝ) (value : Nat)
  | transfer (edge : Transfer) (value : Nat)
  | store (value : Nat) (state : σ) (tensor : Tensor) (index : Nat)

/-- Ordinary selected-object instruction registry at actual execution
occurrences. `active` is the dynamic control/launch selection, not a numerical
postcondition. The finite graph below names only executed instructions. -/
structure InstructionRegistry (σ : Type) where
  opcode : Occurrence σ → Option (Opcode σ)
  output : Occurrence σ → Nat
  active : Occurrence σ → Prop

def LocatedGraph.output (registry : InstructionRegistry σ) : LocatedGraph σ 0 → Nat
  | .input i => Fin.elim0 i
  | .load site _ _ _ _ | .raw site _ | .literal site _ | .natural site _ | .negativeInfinity site => registry.output site
  | .binary site _ _ _ | .unary site _ _ | .fma site _ _ _ | .hostScale site _ _ | .transfer site _ _ => registry.output site

/-- Binding is an opcode/operand/address check against the actual selected
object. The only value premise is one input word's ordinary load observation.
There are no intermediate results or reduction equalities among its fields. -/
def LocatedGraph.Bound (registry : InstructionRegistry σ) (readWord : σ → Tensor → Nat → Word) :
    LocatedGraph σ 0 → Prop
  | .input i => Fin.elim0 i
  | .load site state tensor index x => registry.active site ∧ registry.opcode site=some (.load state tensor index) ∧ readWord state tensor index=x
  | .raw site x => registry.active site ∧ registry.opcode site=some (.raw x)
  | .literal site x => registry.active site ∧ registry.opcode site=some (.literal x)
  | .natural site x => registry.active site ∧ registry.opcode site=some (.natural x)
  | .negativeInfinity site => registry.active site ∧ registry.opcode site=some .negativeInfinity
  | .binary site op a b => registry.active site ∧
      registry.opcode site=some (.binary op (a.output registry) (b.output registry)) ∧ a.Bound registry readWord ∧ b.Bound registry readWord
  | .unary site op a => registry.active site ∧
      registry.opcode site=some (.unary op (a.output registry)) ∧ a.Bound registry readWord
  | .fma site a b c => registry.active site ∧
      registry.opcode site=some (.fma (a.output registry) (b.output registry) (c.output registry)) ∧
      a.Bound registry readWord ∧ b.Bound registry readWord ∧ c.Bound registry readWord
  | .hostScale site scale a => registry.active site ∧
      registry.opcode site=some (.hostScale scale (a.output registry)) ∧ a.Bound registry readWord
  | .transfer site edge a => registry.active site ∧
      registry.opcode site=some (.transfer edge (a.output registry)) ∧ a.Bound registry readWord

/-- The platform executes each active opcode using its named operands. These
are fixed-arity instruction interfaces, independent of the task/source graph. -/
structure InstructionExecution {σ : Type} (registry : InstructionRegistry σ) (trace : PrimitiveTrace σ) : Prop where
  raw : ∀ site x, registry.active site → registry.opcode site=some (.raw x) → trace.raw site x (registry.output site)
  load : ∀ site state tensor index, registry.active site → registry.opcode site=some (.load state tensor index) →
    trace.load site state tensor index (registry.output site)
  literal : ∀ site x, registry.active site → registry.opcode site=some (.literal x) → trace.literal site x (registry.output site)
  natural : ∀ site x, registry.active site → registry.opcode site=some (.natural x) → trace.natural site x (registry.output site)
  negativeInfinity : ∀ site, registry.active site → registry.opcode site=some .negativeInfinity → trace.negativeInfinity site (registry.output site)
  binary : ∀ site op a b, registry.active site → registry.opcode site=some (.binary op a b) → trace.binary site op a b (registry.output site)
  unary : ∀ site op a, registry.active site → registry.opcode site=some (.unary op a) → trace.unary site op a (registry.output site)
  fma : ∀ site a b c, registry.active site → registry.opcode site=some (.fma a b c) → trace.fma site a b c (registry.output site)
  hostScale : ∀ site scale a, registry.active site → registry.opcode site=some (.hostScale scale a) → trace.hostScale site scale a (registry.output site)
  transfer : ∀ site edge a, registry.active site → registry.opcode site=some (.transfer edge a) → trace.transfer site edge a (registry.output site)
  store : ∀ site value state tensor index, registry.active site → registry.opcode site=some (.store value state tensor index) →
    trace.store site value state tensor index

theorem LocatedGraph.executed {σ : Type} {registry : InstructionRegistry σ} {trace : PrimitiveTrace σ}
    (api : InstructionExecution registry trace) {readWord} (graph : LocatedGraph σ 0)
    (binding : graph.Bound registry readWord) :
    LocatedExecution trace readWord Fin.elim0 graph (graph.output registry) := by
  induction graph with
  | input i => exact Fin.elim0 i
  | load site state tensor index x => exact .loaded site state tensor index x _ (api.load _ _ _ _ binding.1 binding.2.1) binding.2.2
  | raw site x => exact .raw site x _ (api.raw _ _ binding.1 binding.2)
  | literal site x => exact .literal site x _ (api.literal _ _ binding.1 binding.2)
  | natural site x => exact .natural site x _ (api.natural _ _ binding.1 binding.2)
  | negativeInfinity site => exact .negativeInfinity site _ (api.negativeInfinity _ binding.1 binding.2)
  | binary site op a b ia ib =>
    exact .binary site op a b _ _ _ (ia binding.2.2.1) (ib binding.2.2.2)
      (api.binary _ _ _ _ binding.1 binding.2.1)
  | unary site op a ih => exact .unary site op a _ _ (ih binding.2.2) (api.unary _ _ _ binding.1 binding.2.1)
  | fma site a b c ia ib ic =>
    exact .fma site a b c _ _ _ _ (ia binding.2.2.1) (ib binding.2.2.2.1) (ic binding.2.2.2.2)
      (api.fma _ _ _ _ binding.1 binding.2.1)
  | hostScale site scale a ih => exact .hostScale site scale a _ _ (ih binding.2.2) (api.hostScale _ _ _ binding.1 binding.2.1)
  | transfer site edge a ih => exact .transfer site edge a _ _ (ih binding.2.2) (api.transfer _ _ _ binding.1 binding.2.1)

structure LinkedCode {σ : Type} {selection : Selection σ} {expression : Expr}
    (plan : SelectedPlan selection expression) (registry : InstructionRegistry σ)
    (readWord : σ → Tensor → Nat → Word) where
  nativeGraph : LocatedGraph σ 0
  lowering : LocalRewrite nativeGraph.erase plan.compile.value.erase
  origins : ∀ site ∈ plan.compile.value.sites,
    ∃ actual ∈ nativeGraph.sites, actual.call=site.call ∧ actual.coordinate=site.coordinate
  binding : nativeGraph.Bound registry readWord

def LinkedCode.executed {σ : Type} {selection : Selection σ} {expression : Expr}
    {plan : SelectedPlan selection expression} {registry : InstructionRegistry σ} {readWord trace}
    (code : LinkedCode plan registry readWord) (api : InstructionExecution registry trace) :
    CompiledExecution plan trace readWord :=
  ⟨code.nativeGraph,code.lowering,code.origins,code.nativeGraph.output registry,code.nativeGraph.executed api code.binding⟩

#axiomsClean LocatedGraph.executed
#axiomsClean LinkedCode.executed
#specNonCircular InstructionExecution avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
