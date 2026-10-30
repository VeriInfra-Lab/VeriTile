import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.ResidentLanguage
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Origins

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Resident
open Host Floating
open CrossEntropyOps.Resident (ResidentKey InstalledCompiler Representations ResidentRefinement)
noncomputable section
set_option maxHeartbeats 5000000

structure Tape {σ : Type} (physical : Physical.Layout σ) where
  initialNative : σ
  final : Origins
  events : List Event
  sourceSteps : OriginRun (Physical.rawWord physical initialNative) emptyOrigins events final

/-- Numeric storage uses four input bytes and two FP16 output bytes; the
instruction accumulator graph remains FP32. Every computed cell uses its
current, already chosen reaching definition and fixed instruction plan. -/
def NumericRepresentation {σ : Type} (physical : Physical.Layout σ) (numerics : Numerics)
    (tape : Tape physical) (source : World Expr σ) : Prop :=
  ∀ cell,(∃ word,source.memory cell=.raw word) ∨
    ∃ id node,tape.final.cells cell=some id ∧ tape.final.nodes id=some node ∧
      node.expression=source.memory cell ∧
      physical.unit cell.1*(cell.2+1)≤physical.capacity source.native cell.1 ∧
      ∀ lane<physical.unit cell.1,
        physical.observe source.native cell.1 (physical.unit cell.1*cell.2+lane)=
          Physical.wordByte (node.plan.graph.eval numerics) lane

/-- Explicit ordinary compiler/frontend/runtime boundary for the actual
installed source and artifact. Labels bind actual ISA operands, fragment
order, async fences/commits/waits, source visits/masks/addresses, complete
ABI and allocation history to the same origin tape. The finite ISA catalog
is explicit in Floating.ISA; no other target lowering is claimed here.
This interface contains no mathematical matrix product or result oracle. -/
structure Engine (σ : Type) where
  physical : Physical.Layout σ
  live : σ → Nat → Prop
  native : Numerics
  Artifact : Type
  NativeTrace : Type
  resident : (program : Arguments) → ResidentKey (language σ) program → Artifact → Prop
  primitives : Model σ → Artifact → Prop
  execute : Arguments → Artifact → CompleteABI σ → σ → Tensor → σ → NativeTrace → Tape physical → Prop
  labels : (model : Model σ) → (program : Arguments) → ResidentKey (language σ) program → Artifact → NativeTrace →
    (initial : World (Scalar model.mode) σ) → (result : Tensor) → (final : World (Scalar model.mode) σ) →
    run model program initial=.ok result final → Tape physical → Prop

structure Engine.Trace {σ : Type} (engine : Engine σ) where
  native : engine.NativeTrace
  tape : Tape engine.physical

def Engine.compiler {σ : Type} (engine : Engine σ) : InstalledCompiler (language σ) where
  Artifact := engine.Artifact
  Trace := engine.Trace
  resident := engine.resident
  models model artifact := engine.primitives model artifact ∧ model.live=engine.live ∧
    Host.Library model.runtime engine.live ∧ Physical.Library model.runtime engine.physical
  execute program artifact abi initial result final trace :=
    engine.execute program artifact abi initial result final trace.native trace.tape ∧
    trace.tape.initialNative=initial ∧ ∀ event∈trace.tape.events,event.site.call<abi.calls.length

def initialRepresentation {σ : Type} (engine : Engine σ) : (mode : Interpretation) → World (Scalar mode) σ → Prop
  | .numeric,source => source.memory=(fun cell => .raw (Physical.rawWord engine.physical source.native cell))
  | .marker,source => source.memory=Physical.initialMarkers

def finalRepresentation {σ : Type} (engine : Engine σ) (trace : engine.Trace) :
    (mode : Interpretation) → World (Scalar mode) σ → World (Scalar mode) σ → Prop
  | .numeric,_,final => NumericRepresentation engine.physical engine.native trace.tape final
  | .marker,initial,final => Physical.Representation engine.physical initial.native final.native final.memory

def Engine.representations {σ : Type} (engine : Engine σ) : Representations (language σ) engine.compiler where
  initial model _ _ _ _ initial := initialRepresentation engine model.mode initial
  final model program artifact abi trace initial result final :=
    ∃ success : run model program initial=.ok result final,
      engine.labels model program ((language σ).key program abi) artifact trace.native initial result final success trace.tape ∧
      finalRepresentation engine trace model.mode initial final

structure CompilerCorrect {σ : Type} (engine : Engine σ) : Prop where
  refinement : ResidentRefinement (language σ) engine.compiler engine.representations

def initialWorld {σ : Type} (engine : Engine σ) (native : σ) : (mode : Interpretation) → World (Scalar mode) σ
  | .numeric => ⟨fun cell => .raw (Physical.rawWord engine.physical native cell),native,[]⟩
  | .marker => ⟨Physical.initialMarkers,native,[]⟩

theorem initial_represented {σ : Type} (engine : Engine σ) (native : σ) (mode : Interpretation) :
    initialRepresentation engine mode (initialWorld engine native mode) := by cases mode <;> rfl

def Tape.Admitted {σ : Type} {physical : Physical.Layout σ} (tape : Tape physical)
    (numerics : Numerics) (domain : Domain) : Prop :=
  ∀ id node,tape.final.nodes id=some node → node.plan.graph.Admitted numerics domain

/-- Transport of the existing reaching plan; no post-hoc plan selection.
All numerical steps below are scalar cast/comparison/ordered multiplication
or fixed ISA applications. -/
theorem NumericRepresentation.reference {σ : Type} {physical : Physical.Layout σ} {native reference : Numerics}
    {tape : Tape physical} {source : World Expr σ} (rep : NumericRepresentation physical native tape source)
    (domain : Domain) (ports : LocalPorts native reference domain) (admitted : tape.Admitted native domain)
    (cell : Address) (expression : Expr) (sourceValue : source.memory cell=expression)
    (computed : ∀ word,expression≠.raw word) :
    ∃ id node,tape.final.cells cell=some id ∧ tape.final.nodes id=some node ∧
      ∃ same : node.expression=expression,
      physical.unit cell.1*(cell.2+1)≤physical.capacity source.native cell.1 ∧
      ∀ lane<physical.unit cell.1,
        physical.observe source.native cell.1 (physical.unit cell.1*cell.2+lane)=
          Physical.wordByte ((node.transport same).graph.sourceEval reference) lane := by
  rcases rep cell with raw|⟨id,node,reaching,found,meaning,bound,bytes⟩
  · obtain ⟨word,same⟩ := raw
    exact False.elim (computed word (sourceValue.symm.trans same))
  · have same : node.expression=expression := meaning.trans sourceValue
    refine ⟨id,node,reaching,found,same,bound,?_⟩
    intro lane hl
    rw [bytes lane hl,Node.transport_graph,Plan.equivalent node.plan native reference domain ports (admitted id node found)]

#axiomsClean NumericRepresentation.reference
#specNonCircular Engine avoiding [Floating.reference,VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath.formula]
#specNonCircular LocalPorts avoiding [Source.run,Host.run,NumericRepresentation]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Resident
