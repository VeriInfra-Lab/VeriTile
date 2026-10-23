import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentComplete
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentPeers

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Step TensorLibrary IntegerViews CollectiveLibrary SavedPreserves value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Public resident entry with actual participating peer calls. The local
rank aliases the main invocation, all collective metadata agrees, and the
transmitted scalar values are derived from the peers' actual local reductions.
The core theorem supplies all three returned byte families, exact return
identity and Python tuple, captured saved-state continuity and caller frames.
Only ordinary external compiler/runtime/collective and local numeric contracts
remain; no expected peer output or task-output certificate is a premise. -/
theorem complete_resident {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (laws : Peepholes (signNumerics engine.native)) (constants : ConstantPorts (signNumerics engine.native))
    (reference : Numerics) (ports : Obligations (signNumerics engine.native) (signNumerics reference))
    (integer : World SignedExpr σ → Tensor → Nat → Int)
    (call : Entry.Invocation (numericObservations integer) signedFP)
    (ops : TensorLibrary (numericObservations integer) signedHost call.runtime.toRuntime)
    (zeros : ZeroAPI (numericObservations integer) call.runtime signedHost.zero)
    (ints : IntegerViews (numericObservations integer))
    (comms : CollectiveLibrary (numericObservations integer) signedHost call.runtime.toRuntime call.groups.size call.groups.rank)
    (dtype : call.logits.dtype="torch.float32") (labelDtype : call.labels.dtype="torch.int64")
    (messages : PeerMessages call) (compatible : messages.Compatible)
    (inputs : ForwardInputs comms messages.aggregation messages.loss messages.lse)
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
    (((∀ row, row<call.data.rows → ReturnedValue engine.layout engine.observe reference forwardTrace.origins
      call.saved.native call.returned.losses row (finalLoss call.kernel signedHost messages.loss messages.lse row)) ∧
    (∀ row, row<call.data.rows → ReturnedValue engine.layout engine.observe reference forwardTrace.origins
      call.saved.native call.returned.zLosses row (finalZ call.kernel signedHost messages.lse row)) ∧
    (∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
      ReturnedValue engine.layout engine.observe reference backwardTrace.origins after.native output (row*call.data.cols+col)
        (Source.gradientScalar signedFP (call.backwardData versions upstream output zUpstream before after success).parameters
          (integer call.initial call.labels row) (value (numericObservations integer) call.initial call.logits (row*call.data.cols+col))
          (finalLSE call.kernel signedHost messages.lse row) (value (numericObservations integer) before upstream row) col)) ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output zUpstream before after success).first) ∧
    pythonBackward signedFP call.runtime versions call.returned.context call.saved.native upstream zUpstream before=
      .ok [some output,none,none,none,none,none,none,none] after) ∧
    CallerFrames engine.layout engine.observe kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output zUpstream before after success).work output ∧
    CrossEntropy1.Autograd.Unmodified versions
      (CrossEntropy1.Autograd.capture versions call.returned.context.base call.saved.native) before) ∧ messages.SourceValues ∧ messages.Compatible := by
  refine ⟨?_,messages.sourceValues,compatible⟩
  exact complete_resident_core engine correct laws constants reference ports integer call ops zeros ints comms
    dtype labelDtype messages.aggregation messages.loss messages.lse inputs versions upstream output zUpstream before after between
    upstreamLive upstreamShape upstreamDtype success forwardArtifact backwardArtifact forwardABI backwardABI
    forwardTrace backwardTrace installedForward installedBackward primitivesForward primitivesBackward forwardCoherent
    backwardCoherent initial storage continuation nativeForward nativeBackward markerRuntime markerInteger markerGroups
    markerObjects markerGeometry markerRules registered markerForward markerBackward kept forwardLifetime backwardLifetime

#axiomsClean complete_resident
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
