import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.InvocationGraphs
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.InstructionBinding

namespace VeriTile.Bench.Repairs.CrossEntropyOps.ConfiguredPlan
open CrossEntropy1.Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Fusion is selected at one source occurrence with the complete actual
request. Pointer alignment, scalar words, compiler and before-state remain
observable in the selection; no block-width-only policy is present. -/
structure FusionSelection (σ : Type) where
  selected : InvocationGraphs.Request σ → List Nat → Nat → Bool → Prop

structure Policy {σ : Type} (selection : FusionSelection σ) where
  request : CallDescriptor σ → InvocationGraphs.Request σ
  contract : Occurrence σ → Bool
  chosen : ∀ site, selection.selected (request site.call) site.coordinate site.instruction (contract site)
  before : ∀ call, (request call).before=call.before
  invocation : ∀ call, (request call).invocation=call.invocation
  stream : ∀ call, (request call).stream=call.stream
  rank : ∀ call, (request call).rank=call.rank

/-- Task-independent compilation of the source-indexed occurrence plan,
reusing the exact selected reduction trees. Every add/sub may preserve its
rounded operands or contract the immediately adjacent product. Load barriers
and host calls always end the opportunity. -/
def compile {σ : Type} {platform : Selection σ} {expression : Expr}
    (contract : Occurrence σ → Bool) :
    SelectedPlan platform expression → LocatedCompiled σ
  | .raw x origin => ⟨match origin with
      | .constant site => .raw site x
      | .load site state tensor index => .load site state tensor index x,none⟩
  | .literal x site => ⟨.literal site x,none⟩
  | .natural x site => ⟨.natural site x,none⟩
  | .negativeInfinity site => ⟨.negativeInfinity site,none⟩
  | .barrier site edge x => ⟨.transfer site edge (compile contract x).value,none⟩
  | .binary op site a b =>
    let left := (compile contract a)
    let right := (compile contract b)
    match op with
    | .add => if contract site then emitLocatedAdd site left right else ⟨.binary site .add left.value right.value,none⟩
    | .sub => if contract site then emitLocatedSub site left right else ⟨.binary site .sub left.value right.value,none⟩
    | .mul => ⟨.binary site .mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨.binary site .div left.value right.value,none⟩
  | .unary op site x => ⟨.unary site (match op with | .exp => .exp | .log => .log) (compile contract x).value,none⟩
  | .sum choice xs => ⟨(choice.graph .add).bind (fun i => (compile contract (xs i)).value),none⟩
  | .maximum choice xs => ⟨(choice.graph .maximum).bind (fun i => (compile contract (xs i)).value),none⟩
  | .negation site x => ⟨.unary site .neg (compile contract x).value,none⟩
  | .hostAdd site a b => ⟨.binary site .hostAdd (compile contract a).value (compile contract b).value,none⟩
  | .hostSquare site x => ⟨.unary site .hostSquare (compile contract x).value,none⟩
  | .hostScale scale site x => ⟨.hostScale site scale (compile contract x).value,none⟩
  | .hostSum choice xs => ⟨(choice.graph .hostAdd).bind (fun i => (compile contract (xs i)).value),none⟩
  | .hostLSE maxChoice sumChoice maskSite logSite addSite zeroSite subSite expSite xs =>
    ⟨hostLSELocated maxChoice sumChoice maskSite logSite addSite zeroSite subSite expSite
      (fun i => (compile contract (xs i)).value),none⟩
  | .rankSum _ choice xs => ⟨(choice.graph .rankAdd).bind (fun i => (compile contract (xs i)).value),none⟩

def reference {σ : Type} {platform : Selection σ} {expression : Expr}
    (contract : Occurrence σ → Bool) (plan : SelectedPlan platform expression) (numerics : Numerics) : Word :=
  (compile contract plan).value.erase.eval numerics Fin.elim0

theorem reference_equiv {σ : Type} {platform : Selection σ} {expression : Expr}
    (contract : Occurrence σ → Bool) (plan : SelectedPlan platform expression)
    (native expected : Numerics) (ports : Obligations native expected) :
    reference contract plan native=reference contract plan expected :=
  (compile contract plan).value.erase.equiv ports Fin.elim0 Fin.elim0 (fun i => Fin.elim0 i)

/-- Only local opcode rewrites may relate decoded code to the configured
source plan. There is no FMA-versus-multiply/add equality constructor. -/
structure LinkedCode {σ : Type} {platform : Selection σ} {expression : Expr}
    (contract : Occurrence σ → Bool) (plan : SelectedPlan platform expression)
    (registry : InstructionRegistry σ) (readWord : σ → Tensor → Nat → Word) where
  graph : LocatedGraph σ 0
  lowering : LocalRewrite graph.erase (compile contract plan).value.erase
  origins : ∀ site ∈ (compile contract plan).value.sites,
    ∃ actual ∈ graph.sites, actual.call=site.call ∧ actual.coordinate=site.coordinate
  binding : graph.Bound registry readWord

theorem LinkedCode.reference_value {σ : Type} {platform : Selection σ} {expression : Expr}
    {contract : Occurrence σ → Bool} {plan : SelectedPlan platform expression}
    {registry : InstructionRegistry σ} {readWord readByte} {trace : PrimitiveTrace σ}
    (code : LinkedCode contract plan registry readWord) (execution : InstructionExecution registry trace)
    (native expected : Numerics) (api : PrimitiveContracts trace native readWord readByte)
    (ports : Obligations native expected) (laws : Peepholes native) (constants : ConstantPorts native) :
    trace.bits (code.graph.output registry)=reference contract plan expected := by
  have run := code.graph.executed execution code.binding
  exact (run.erase.native_value api).trans
    ((code.lowering.value laws constants Fin.elim0).trans (reference_equiv contract plan native expected ports))

theorem barrier_stops_contraction {σ : Type} {platform : Selection σ} {x : Expr}
    (contract : Occurrence σ → Bool) (site : Occurrence σ) (edge : Transfer) (value : SelectedPlan platform x) :
    (compile contract (.barrier site edge value)).product=none := rfl

#axiomsClean reference_equiv
#axiomsClean LinkedCode.reference_value
#axiomsClean barrier_stops_contraction
#specNonCircular FusionSelection avoiding [Source.forward,Source.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.ConfiguredPlan
