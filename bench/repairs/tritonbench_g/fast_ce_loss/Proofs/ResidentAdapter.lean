import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.ResidentLanguage
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Origins
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Input

namespace VeriTile.Bench.Repairs.FastCeLoss.Resident
open Source Host Floating
open CrossEntropyOps.Resident (ResidentKey InstalledCompiler Representations ResidentRefinement)
noncomputable section
set_option maxHeartbeats 5000000

/-- Fixed little-endian byte observations. The integer channel reads two words
at the actual int64 coordinate; it is not an independently supplied label array. -/
def rawWord {σ : Type} (physical : Physical.Layout σ) (native : σ) (cell : Address) : Word :=
  BitVec.ofNat 32 (∑ lane : Fin 4,(physical.observe native cell.1 (4*cell.2+lane.val)).toNat*256^lane.val)

def rawInteger {σ : Type} (physical : Physical.Layout σ) (native : σ) (cell : Address) : Int :=
  ((rawWord physical native (cell.1,2*cell.2+1)).append (rawWord physical native (cell.1,2*cell.2))).toInt

def wordByte (word : Word) (lane : Fin 4) : Physical.Byte :=
  BitVec.ofNat 8 (word.toNat/256^lane.val)

def emptyOrigins : Origins := ⟨0,fun _ => none,fun _ => none⟩

structure Tape {σ : Type} (physical : Physical.Layout σ) where
  initialNative : σ
  final : Origins
  events : List Event
  sourceSteps : OriginRun (rawWord physical initialNative) emptyOrigins events final

/-- Every numerical stored value keeps the fixed current reaching ID of this
execution. The raw alternative is for unchanged/unmaterialized memory only.
It cannot represent a computed loss, LSE or gradient source expression. -/
def NumericRepresentation {σ : Type} (physical : Physical.Layout σ) (numerics : Numerics)
    (tape : Tape physical) (source : World Expr σ) : Prop :=
  ∀ cell,(∃ word,source.memory cell=.raw word) ∨
    ∃ id node,tape.final.cells cell=some id ∧ tape.final.nodes id=some node ∧
      node.expression=source.memory cell ∧ 4*(cell.2+1)≤physical.capacity source.native cell.1 ∧
      ∀ lane : Fin 4,physical.observe source.native cell.1 (4*cell.2+lane.val)=
        wordByte (node.plan.graph.eval numerics) lane

/-- Ordinary frontend/compiler/library boundary for one actual artifact and
execution. Its source-labelled trace must retain exact AST paths/visits,
ordered scalar operands (including commuted native multiplications), all
nested reduction and fusion selections, actual masks/addresses and allocator
history. Raw Python cap/scale bits are bound through InputContract to Options,
and raw int64 labels share these same native bytes. Registered descriptor
pointers/devices/versions and library scalar conversion are part of primitives.
This is an explicit external implementation boundary, not a constructed
compiler proof or a numerical whole-output hypothesis. -/
structure Engine (σ : Type) where
  physical : Physical.Layout σ
  live : σ → Nat → Prop
  native : Numerics
  Artifact : Type
  NativeTrace : Type
  resident : (program : Program) → ResidentKey (language σ) program → Artifact → Prop
  primitives : Model σ → Artifact → Prop
  execute : (program : Program) → Artifact → CompleteABI σ → σ → WholeReturn → σ →
    NativeTrace → Tape physical → Prop
  labels : (model : Model σ) → (program : Program) → ResidentKey (language σ) program → Artifact → NativeTrace →
    (initial : World (Scalar model.interpretation) σ) → (result : WholeReturn) →
    (final : World (Scalar model.interpretation) σ) → run model program initial=.ok result final → Tape physical → Prop

structure Engine.Trace {σ : Type} (engine : Engine σ) where
  native : engine.NativeTrace
  tape : Tape engine.physical

def Engine.compiler {σ : Type} (engine : Engine σ) : InstalledCompiler (language σ) where
  Artifact := engine.Artifact
  Trace := engine.Trace
  resident := engine.resident
  models model artifact := engine.primitives model artifact ∧ Library model.runtime engine.live
  execute program artifact abi initial result final trace :=
    engine.execute program artifact abi initial result final trace.native trace.tape ∧
    trace.tape.initialNative=initial ∧
    ∀ event∈trace.tape.events,event.site.request<abi.calls.length

def initialRepresentation {σ : Type} (engine : Engine σ) : (mode : Interpretation) → World (Scalar mode) σ → Prop
  | .numeric,source => source.memory=(fun cell => .raw (rawWord engine.physical source.native cell)) ∧
      source.integer=rawInteger engine.physical source.native
  | .marker,source => source.memory=Physical.initialMarkers ∧ source.integer=rawInteger engine.physical source.native

def finalRepresentation {σ : Type} (engine : Engine σ) (trace : engine.Trace) :
    (mode : Interpretation) → World (Scalar mode) σ → World (Scalar mode) σ → Prop
  | .numeric,_,final => NumericRepresentation engine.physical engine.native trace.tape final
  | .marker,initial,final => Physical.Representation engine.physical initial.native final.native final.memory

def Engine.representations {σ : Type} (engine : Engine σ) : Representations (language σ) engine.compiler where
  initial model _ _ _ _ initial := initialRepresentation engine model.interpretation initial
  final model program artifact abi trace initial result final :=
    ∃ success : run model program initial=.ok result final,
      engine.labels model program ((language σ).key program abi) artifact trace.native initial result final success trace.tape ∧
      finalRepresentation engine trace model.interpretation initial final

structure CompilerCorrect {σ : Type} (engine : Engine σ) : Prop where
  refinement : ResidentRefinement (language σ) engine.compiler engine.representations

/-- Initial numeric and marker states use the same actual native bytes and
runtime. Only their fixed scalar interpretations differ. -/
def initialWorld {σ : Type} (engine : Engine σ) (native : σ) :
    (mode : Interpretation) → World (Scalar mode) σ
  | .numeric => ⟨fun cell => .raw (rawWord engine.physical native cell),rawInteger engine.physical native,native,[],[]⟩
  | .marker => ⟨Physical.initialMarkers,rawInteger engine.physical native,native,[],[]⟩

theorem initial_represented {σ : Type} (engine : Engine σ) (native : σ) (mode : Interpretation) :
    initialRepresentation engine mode (initialWorld engine native mode) := by cases mode <;> exact ⟨rfl,rfl⟩

def Tape.Admitted {σ : Type} {physical : Physical.Layout σ} (tape : Tape physical)
    (numerics : Numerics) (domain : Domain) : Prop :=
  ∀ id node,tape.final.nodes id=some node → node.plan.graph.Admitted numerics domain

/-- Transport follows the fixed reaching definition. No plan is selected after
matching the desired source expression. All deferred ports are scalar calls. -/
theorem NumericRepresentation.reference {σ : Type} {physical : Physical.Layout σ} {native reference : Numerics}
    {tape : Tape physical} {source : World Expr σ} (rep : NumericRepresentation physical native tape source)
    (domain : Domain) (ports : LocalPorts native reference domain) (admitted : tape.Admitted native domain)
    (cell : Address) (expression : Expr) (sourceValue : source.memory cell=expression)
    (computed : ∀ word,expression≠.raw word) :
    ∃ id node,tape.final.cells cell=some id ∧ tape.final.nodes id=some node ∧
      ∃ same : node.expression=expression,
      4*(cell.2+1)≤physical.capacity source.native cell.1 ∧
      ∀ lane : Fin 4,physical.observe source.native cell.1 (4*cell.2+lane.val)=
        wordByte ((node.transport same).graph.eval reference) lane := by
  rcases rep cell with raw|⟨id,node,reaching,found,meaning,bound,bytes⟩
  · obtain ⟨word,same⟩ := raw
    exact False.elim (computed word (sourceValue.symm.trans same))
  · have same : node.expression=expression := meaning.trans sourceValue
    refine ⟨id,node,reaching,found,same,bound,?_⟩
    intro lane
    rw [bytes lane,Node.transport_word,Plan.equivalent node.plan native reference domain ports (admitted id node found)]

#axiomsClean initial_represented
#axiomsClean NumericRepresentation.reference
#specNonCircular Engine avoiding [Math.loss,Math.gradient,Complete.lseGraph,Complete.scalarGraph,API.gradientGraph]
#specNonCircular NumericRepresentation avoiding [Math.loss,Math.gradient,Complete.lseGraph,Complete.scalarGraph,API.gradientGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Resident
