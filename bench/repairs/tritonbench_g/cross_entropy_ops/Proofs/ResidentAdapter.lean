import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentLanguage
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NumericRepresentation
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.MarkerPrimitives

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Step TensorLibrary IntegerViews CollectiveLibrary Groups ObjectRegistry ViewGeometry)
open CrossEntropy1.Physical (Layout Byte)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def allocationRules {σ : Type} : (mode : Interpretation) → Host.Runtime (Scalar mode) σ → Prop
  | .numeric, _ => True
  | .origins, rt => MarkerAllocationRules rt

/-- The source model is admitted only through ordinary primitive laws under
the two fixed arithmetic and sign interpretations. These are not fields for
a whole forward/backward result or physical caller frame. -/
structure PrimitiveRules {σ : Type} (model : Model σ) where
  library : Host.LocalAPI model.observations model.runtime
  arithmetic : TensorLibrary model.observations (hostInterpreter model.mode) model.runtime.toRuntime
  zeros : Host.ZeroAPI model.observations model.runtime (hostInterpreter model.mode).zero
  integers : IntegerViews model.observations
  collectives : CollectiveLibrary model.observations (hostInterpreter model.mode)
    model.runtime.toRuntime model.groupSize model.groupRank
  groups : Groups model.observations model.runtime.toRuntime
  objects : ObjectRegistry model.runtime.toRuntime
  geometry : ViewGeometry model.runtime.toRuntime
  allocations : allocationRules model.mode model.runtime

/-- Source/native correspondence for a snapshot before execution. A cell
without a previous source definition can only be an actual raw input word.
A saved defined cell retains its exact node and sign; it cannot be reselected
by expression equality. Establishing a backward snapshot must reuse the
forward snapshot through ordinary saved-storage/version preservation. -/
structure NumericInitial {σ : Type} {selection : Selection σ} {fusion}
    (layout : Layout σ) (observe : σ → Nat → Nat → Byte) (native : Numerics)
    (trace : OriginTrace selection fusion (rawWordAt layout observe)) (source : World SignedExpr σ) : Prop where
  undefined : ∀ r i, source.live r → wordInside layout source.native r i → trace.initial.cells (r,i)=none →
    (source.memory r i).read .real=⟨.raw (rawWordAt layout observe source.native r i),Bool.false⟩
  defined : ∀ r i stored, source.live r → wordInside layout source.native r i → trace.initial.cells (r,i)=some stored →
    ∃ node, trace.initial.nodes stored.id=some node ∧
      (source.memory r i).read .real=⟨node.expression,stored.negative⟩ ∧
      ∀ byte, rawByte layout observe source.native r i byte=
        CrossEntropy1.Physical.byteOf (storedWord trace.policy.contract node stored.negative native) byte

def MarkerInitial {σ : Type} (source : World Marker σ) : Prop :=
  ∀ r i, (source.memory r i).read .real=Marker.oldCell r i Bool.false

/-- Ordinary installed engine boundary. `execute` records the native witness
AND its one source-level tape jointly. `labels` relates that same witness and
tape to the fixed source run, including full AST/entry/ABI and initial/final
states. It is the explicit trusted frontend boundary; implementing the
compiler or instrumenting Host is not asserted here. No field contains a
cross-entropy formula, selected output plan, or write exclusion.

The external meaning of `labels` includes the actual AST paths and visits,
primitive kinds and ordered operands, masks, addresses, allocation order,
communication routes, and every occurrence embedded inside a tape node.
In particular hostLSE's max/sum/mask/log/add/zero/sub/exp sites and inherited
earlier nodes retain their actual selected object, full compilation request
and fusion decision. `requestsBound` checks only each event's primary site;
it is not a replacement for this ordinary frontend semantic contract.
Passing the key, source success and tape to this abstract relation does not
constitute a Lean construction or verification of that correspondence. -/
structure Engine (σ : Type) where
  layout : Layout σ
  observe : σ → Nat → Nat → Byte
  native : Numerics
  selection : Selection σ
  fusion : ConfiguredPlan.FusionSelection σ
  Artifact : Type
  NativeTrace : Type
  resident : (program : Program σ) → ResidentKey (language σ) program → Artifact → Prop
  primitives : Model σ → Artifact → Prop
  execute : (program : Program σ) → Artifact → CompleteABI program → σ → Result program → σ →
    NativeTrace → OriginTrace selection fusion (rawWordAt layout observe) → Prop
  labels : (model : Model σ) → (program : Program σ) → ResidentKey (language σ) program →
    Artifact → NativeTrace → (initial : World (Scalar model.mode) σ) → (result : Result program) →
    (final : World (Scalar model.mode) σ) → run model program initial=.ok result final →
    OriginTrace selection fusion (rawWordAt layout observe) → Prop

structure Engine.Trace {σ : Type} (engine : Engine σ) where
  native : engine.NativeTrace
  origins : OriginTrace engine.selection engine.fusion (rawWordAt engine.layout engine.observe)

def Engine.compiler {σ : Type} (engine : Engine σ) : InstalledCompiler (language σ) where
  Artifact := engine.Artifact
  Trace := engine.Trace
  resident := engine.resident
  models model artifact := engine.primitives model artifact ∧ Nonempty (PrimitiveRules model)
  execute program artifact abi initial result final trace :=
    engine.execute program artifact abi initial result final trace.native trace.origins ∧
      trace.origins.requestsBound abi.calls

def initialRepresentation {σ : Type} (engine : Engine σ) (trace : engine.Trace) :
    (mode : Interpretation) → World (Scalar mode) σ → Prop
  | .numeric, source => NumericInitial engine.layout engine.observe engine.native trace.origins source
  | .origins, source => MarkerInitial source

def finalRepresentation {σ : Type} (engine : Engine σ) (trace : engine.Trace) (program : Program σ)
    (result : Result program) : (mode : Interpretation) → World (Scalar mode) σ → World (Scalar mode) σ → Prop
  | .numeric, _, final => NumericRep engine.layout engine.observe engine.native trace.origins final (exports program result)
  | .origins, initial, final => MarkerRep engine.layout engine.observe initial.native final

/-- Fixed representations: the final source labels, numeric node meaning and
opaque storage-origin meaning cannot be replaced by a caller-chosen predicate. -/
def Engine.representations {σ : Type} (engine : Engine σ) : Representations (language σ) engine.compiler where
  initial model _ _ _ trace source := initialRepresentation engine trace model.mode source
  final model program artifact abi trace initial result final :=
    ∃ success : run model program initial=.ok result final,
      engine.labels model program ((language σ).key program abi) artifact trace.native initial result final success trace.origins ∧
      finalRepresentation engine trace program result model.mode initial final

/-- External ordinary compiler/runtime correctness, with the unchanged local
peephole and constant obligations made explicit. The reference-numerics ports
are used later by NumericRep.reference_bytes, not hidden here. -/
structure CompilerCorrect {σ : Type} (engine : Engine σ) : Prop where
  refinement : Peepholes (signNumerics engine.native) → ConstantPorts (signNumerics engine.native) →
    ResidentRefinement (language σ) engine.compiler engine.representations

/-- Determinism identifies the source run returned by generic simulation with
the already proved source/host run. This is ordinary execution transport and
does not assume a source-output lowering certificate. -/
theorem resident_run {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (laws : Peepholes (signNumerics engine.native)) (constants : ConstantPorts (signNumerics engine.native))
    (model : Model σ) (program : Program σ) (artifact : engine.Artifact) (abi : CompleteABI program)
    (trace : engine.Trace) (initial final : World (Scalar model.mode) σ) (result : Result program) (nativeFinal : σ)
    (installed : engine.resident program ((language σ).key program abi) artifact)
    (modeled : engine.compiler.models model artifact) (coherent : abi.initial=initial.native)
    (represented : initialRepresentation engine trace model.mode initial)
    (nativeRun : engine.compiler.execute program artifact abi initial.native result nativeFinal trace)
    (sourceRun : run model program initial=.ok result final) :
    final.native=nativeFinal ∧
      kernelMetadata model program initial result final sourceRun=abi.kernel ∧
      engine.representations.final model program artifact abi trace initial result final := by
  obtain ⟨simulated,success,state,metadata,rep⟩ := (correct.refinement laws constants).simulate
    model program artifact abi initial result nativeFinal trace installed modeled coherent represented nativeRun
  have same : simulated=final := (Step.ok.inj (success.symm.trans sourceRun)).2
  subst simulated
  exact ⟨state,metadata,rep⟩

theorem marked_initial {α σ : Type} (source : World α σ) : MarkerInitial (markWorld source) := by
  intro r i
  rfl

#axiomsClean resident_run
#axiomsClean marked_initial
#specNonCircular NumericInitial avoiding [Spec.loss,Spec.zLoss,Spec.gradient]
#specNonCircular Engine avoiding [Spec.loss,Spec.zLoss,Spec.gradient]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
