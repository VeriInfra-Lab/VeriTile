import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentFrames

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Step TensorLibrary IntegerViews CollectiveLibrary SavedPreserves value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Exactly the allocations captured by the actual returned context. -/
def CapturedRegions (context : Host.Context) (r : RegionName) : Prop :=
  ∃ tensor, tensor∈CrossEntropy1.Autograd.savedTensors context.base ∧ r=tensor.region

/-- Core source-to-native composition for the candidate. The same actual
forward and backward calls provide values, exact return identity/ABI and all
caller-byte frames. Between calls, ordinary saved-reference storage evidence
is required on exactly the captured allocations. Successful retrieval supplies
the version-check conclusion; version equality alone never proves bytes.

The only trusted program boundary is the explicit ordinary resident
compiler/frontend/runtime refinement. No task-output lowering certificate,
selected return plan or whole native write exclusion is a premise. -/
theorem complete_resident_core {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (laws : Peepholes (signNumerics engine.native)) (constants : ConstantPorts (signNumerics engine.native))
    (reference : Numerics) (ports : Obligations (signNumerics engine.native) (signNumerics reference))
    (integer : World SignedExpr σ → Tensor → Nat → Int)
    (call : Entry.Invocation (numericObservations integer) signedFP)
    (ops : TensorLibrary (numericObservations integer) signedHost call.runtime.toRuntime)
    (zeros : ZeroAPI (numericObservations integer) call.runtime signedHost.zero)
    (ints : IntegerViews (numericObservations integer))
    (comms : CollectiveLibrary (numericObservations integer) signedHost call.runtime.toRuntime call.groups.size call.groups.rank)
    (dtype : call.logits.dtype="torch.float32") (labelDtype : call.labels.dtype="torch.int64")
    (aggregation : ForwardAggregation call.data) (lossPeers lsePeers : Nat → SignedExpr)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    (versions : σ → Nat → Nat) (upstream output : Tensor) (zUpstream : Option Tensor)
    (before after : World SignedExpr σ)
    (between : SavedPreserves (numericObservations integer) call.returned.context.base call.saved before)
    (upstreamLive : before.live upstream.region) (upstreamShape : upstream.shape=[call.data.rows])
    (upstreamDtype : upstream.dtype="torch.float32")
    (success : backwardEntry signedFP call.runtime versions call.returned.context call.saved.native upstream zUpstream before=.ok output after)
    (forwardArtifact backwardArtifact : engine.Artifact)
    (forwardABI : CompleteABI (.forward call.options call.logits call.labels))
    (backwardABI : CompleteABI (.backward call.returned.context call.saved.native upstream zUpstream))
    (forwardTrace backwardTrace : engine.Trace)
    (installedForward : engine.resident _ ((language σ).key _ forwardABI) forwardArtifact)
    (installedBackward : engine.resident _ ((language σ).key _ backwardABI) backwardArtifact)
    (primitivesForward : engine.primitives (invocationModel call versions) forwardArtifact)
    (primitivesBackward : engine.primitives (invocationModel call versions) backwardArtifact)
    (forwardCoherent : forwardABI.initial=call.initial.native) (backwardCoherent : backwardABI.initial=before.native)
    (initial : NumericInitial engine.layout engine.observe engine.native forwardTrace.origins call.initial)
    (storage : SavedStorage engine.layout engine.observe (CapturedRegions call.returned.context) call.saved before)
    (continuation : OriginContinuation forwardTrace.origins backwardTrace.origins (CapturedRegions call.returned.context))
    (nativeForward : engine.compiler.execute (.forward call.options call.logits call.labels) forwardArtifact forwardABI
      call.initial.native call.returned call.saved.native forwardTrace)
    (nativeBackward : engine.compiler.execute (.backward call.returned.context call.saved.native upstream zUpstream)
      backwardArtifact backwardABI before.native (backwardTuple output) after.native backwardTrace)
    (markerRuntime : Host.Runtime Marker σ)
    (markerInteger : World Marker σ → Tensor → Nat → Int)
    (markerGroups : CrossEntropy1.Host.Groups (markerObservations markerInteger) markerRuntime.toRuntime)
    (markerObjects : CrossEntropy1.Host.ObjectRegistry markerRuntime.toRuntime)
    (markerGeometry : CrossEntropy1.Host.ViewGeometry markerRuntime.toRuntime)
    (markerRules : MarkerPrimitiveRules markerInteger markerRuntime markerGroups.size markerGroups.rank)
    (registered : ∀ t, call.objects.valid t → markerObjects.valid t)
    (markerForward : engine.primitives (markerModel markerRuntime versions markerInteger markerGroups.size markerGroups.rank) forwardArtifact)
    (markerBackward : engine.primitives (markerModel markerRuntime versions markerInteger markerGroups.size markerGroups.rank) backwardArtifact)
    (kept : RegionName → Prop)
    (forwardLifetime : CrossEntropy1.Physical.Retains engine.layout kept call.initial.native call.saved.native)
    (backwardLifetime : CrossEntropy1.Physical.Retains engine.layout kept before.native after.native) :
    ((∀ row, row<call.data.rows → ReturnedValue engine.layout engine.observe reference forwardTrace.origins
      call.saved.native call.returned.losses row (finalLoss call.kernel signedHost lossPeers lsePeers row)) ∧
    (∀ row, row<call.data.rows → ReturnedValue engine.layout engine.observe reference forwardTrace.origins
      call.saved.native call.returned.zLosses row (finalZ call.kernel signedHost lsePeers row)) ∧
    (∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
      ReturnedValue engine.layout engine.observe reference backwardTrace.origins after.native output (row*call.data.cols+col)
        (Source.gradientScalar signedFP (call.backwardData versions upstream output zUpstream before after success).parameters
          (integer call.initial call.labels row) (value (numericObservations integer) call.initial call.logits (row*call.data.cols+col))
          (finalLSE call.kernel signedHost lsePeers row) (value (numericObservations integer) before upstream row) col)) ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output zUpstream before after success).first) ∧
    pythonBackward signedFP call.runtime versions call.returned.context call.saved.native upstream zUpstream before=
      .ok [some output,none,none,none,none,none,none,none] after) ∧
    CallerFrames engine.layout engine.observe kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output zUpstream before after success).work output ∧
    CrossEntropy1.Autograd.Unmodified versions
      (CrossEntropy1.Autograd.capture versions call.returned.context.base call.saved.native) before := by
  have values := complete_resident_values engine correct laws constants reference ports integer call ops zeros ints comms
    dtype labelDtype aggregation lossPeers lsePeers inputs versions upstream output zUpstream before after between
    upstreamLive upstreamShape upstreamDtype success forwardArtifact backwardArtifact forwardABI backwardABI
    forwardTrace backwardTrace installedForward installedBackward primitivesForward primitivesBackward forwardCoherent
    backwardCoherent initial (CapturedRegions call.returned.context) storage continuation nativeForward nativeBackward
  have arrays := forward_graph_values (numericObservations integer) signedHost call.runtime call.library ops zeros ints call.objects
    call.groups.size call.groups.rank comms call.kernel call.views labelDtype call.size aggregation lossPeers lsePeers inputs
  have context := marked_context (between.context arrays.2.2.2.context) markerRuntime markerObjects registered
  let model := invocationModel call versions
  let rules : PrimitiveRules model :=
    ⟨call.library,ops,zeros,ints,comms,call.groups,call.objects,call.geometry,True.intro⟩
  have forward := resident_run engine correct laws constants model _ forwardArtifact forwardABI forwardTrace
    call.initial call.saved call.returned call.saved.native installedForward ⟨primitivesForward,⟨rules⟩⟩
    forwardCoherent initial nativeForward call.success
  obtain ⟨forwardSuccess,forwardLabels,forwardRep⟩ := forward.2.2
  have forwardRep : NumericRep engine.layout engine.observe engine.native forwardTrace.origins call.saved
      (exports (.forward call.options call.logits call.labels) call.returned) := forwardRep
  have nextInitial := NumericInitial.from_saved forwardRep storage continuation
  have python := python_return signedFP call.runtime versions _ _ upstream output zUpstream before after success
  have backward := resident_run engine correct laws constants model _ backwardArtifact backwardABI backwardTrace
    before after (backwardTuple output) after.native installedBackward ⟨primitivesBackward,⟨rules⟩⟩
    backwardCoherent nextInitial nativeBackward python
  let b := call.backwardData versions upstream output zUpstream before after success
  have workBinding : backwardABI.kernel.outputs=[b.work] :=
    (congrArg KernelABI.outputs backward.2.1).symm.trans
      (backward_metadata_work model call.returned.context call.saved.native upstream output zUpstream before after success)
  refine ⟨values,?_,(backward_entry_success signedFP call.runtime versions _ _ upstream output zUpstream before after success).1⟩
  exact resident_marker_frames engine correct laws constants markerRuntime versions markerInteger markerGroups markerObjects
    markerGeometry markerRules call.options call.logits call.labels call.returned call.initial call.saved before after
    upstream output b.work zUpstream call.logitsLive call.labelsLive call.rank (registered _ call.registered) labelDtype
    context upstreamLive forwardArtifact backwardArtifact forwardABI backwardABI forwardTrace backwardTrace installedForward
    installedBackward markerForward markerBackward forwardCoherent backwardCoherent nativeForward nativeBackward workBinding
    kept forwardLifetime backwardLifetime

#axiomsClean complete_resident_core
#specNonCircular CapturedRegions avoiding [Spec.loss,Spec.zLoss,Spec.gradient]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
