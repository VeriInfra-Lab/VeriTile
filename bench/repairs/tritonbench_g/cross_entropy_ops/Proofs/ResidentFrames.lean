import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentValues
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.MarkerFrames

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Step TensorLibrary IntegerViews CollectiveLibrary SavedPreserves BackwardContext)
open CrossEntropy1.Physical (Layout Byte Retains OutsideView)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- The physical caller-frame conclusion is scalar-independent. It observes
all allocation bytes, including view holes and partial final words. -/
structure CallerFrames {α σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (kept : RegionName → Prop) (initial saved before after : World α σ)
    (inplace : Bool) (work output : Tensor) : Prop where
  forward : ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
    offset<(layout.buffer initial.native r).capacity →
    observe saved.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)
  backward : ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity → OutsideView work r offset → OutsideView output r offset →
    observe after.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      observe before.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)
  outOfPlace : inplace≠Bool.true → ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity →
    observe after.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      observe before.native (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)

def markerModel {σ : Type} (rt : Host.Runtime Marker σ) (versions : σ → Nat → Nat)
    (integer : World Marker σ → Tensor → Nat → Int) (gs gr : Nat → Nat) : Model σ :=
  ⟨.origins,rt,versions,integer,gs,gr⟩

theorem actual_rank_inputs {α σ : Type} [Inhabited α]
    {obs : CrossEntropy1.Host.Observations α σ} {A : CrossEntropy1.Host.TensorArithmetic α}
    {rt : CrossEntropy1.Host.Runtime α σ} {gs gr : Nat → Nat}
    (comms : CollectiveLibrary obs A rt gs gr)
    {o : CrossEntropy1.Host.Options} {world rows : Nat} {localPair pair : Tensor × Tensor}
    {initial final : World α σ} (trace : CrossEntropy1.Host.RankTrace rt o world rows localPair initial pair final) :
    ∃ lossPeers lsePeers, CrossEntropy1.Host.RankInputs comms trace lossPeers lsePeers := by
  cases trace with
  | direct one => exact ⟨fun _ => default,fun _ => default,True.intro⟩
  | reduced many gathered globalLSE handle allocated collected pending reduced waited allocationCall gatherCall sumCall lseCall waitCall =>
    exact ⟨_,_,⟨fun _ _ => rfl,fun _ _ => rfl⟩⟩

/-- Frames use the actual collective input functions, with no assertion
about peer arithmetic results. -/
theorem actual_forward_inputs {α σ : Type} [Inhabited α]
    {obs : CrossEntropy1.Host.Observations α σ} {A : CrossEntropy1.Host.TensorArithmetic α}
    {rt : Host.Runtime α σ} {gs gr : Nat → Nat}
    (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : CrossEntropy1.Source.FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (a : ForwardAggregation e) :
    ∃ lossPeers lsePeers, ForwardInputs comms a lossPeers lsePeers := by
  cases a with
  | direct flag => exact ⟨fun _ => default,fun _ => default,True.intro⟩
  | split flag trace =>
    exact actual_rank_inputs comms trace.rankCalls

/-- Only ordinary registered-object metadata crosses scalar interpretations. -/
theorem marked_context {α σ : Type} {rt : CrossEntropy1.Host.Runtime α σ}
    {objects : CrossEntropy1.Host.ObjectRegistry rt} {c : CrossEntropy1.Host.Context} {before : World α σ}
    (ctx : BackwardContext rt objects c before)
    (markerRuntime : Host.Runtime Marker σ)
    (markerObjects : CrossEntropy1.Host.ObjectRegistry markerRuntime.toRuntime)
    (registered : ∀ t, objects.valid t → markerObjects.valid t) :
    BackwardContext markerRuntime.toRuntime markerObjects c (markWorld before) :=
  ⟨ctx.logitsLive,ctx.labelsLive,ctx.lseLive,ctx.destinationLive,ctx.logitsRank,ctx.logitsNotNeg,
    registered _ ctx.logitsObject,registered _ ctx.destinationObject,ctx.destinationShape,
    ctx.logitsLabels,ctx.logitsLSE,ctx.rows⟩

theorem backward_metadata_work {σ : Type} (model : Model σ) (c : Context) (saved : σ)
    (upstream output : Tensor) (z : Option Tensor)
    (before after : World (Scalar model.mode) σ)
    (success : backwardEntry (scalarInterpreter model.mode) model.runtime model.versions c saved upstream z before=.ok output after) :
    (kernelMetadata model (.backward c saved upstream z) before (backwardTuple output) after
      (python_return _ _ _ _ _ _ _ _ _ _ success)).outputs=
      [(Classical.choice (backward_success (scalarInterpreter model.mode) model.runtime c upstream output z before after
        (backward_entry_success _ _ _ _ _ _ _ _ _ _ success).2)).work] := by
  let h := python_success (scalarInterpreter model.mode) model.runtime model.versions c saved upstream z
    before after (backwardTuple output) (python_return _ _ _ _ _ _ _ _ _ _ success)
  have same : Classical.choose h=output := by
    have tuple := (Classical.choose_spec h).2
    simpa only [backwardTuple,List.cons.injEq,Option.some.injEq,and_true] using tuple.symm
  simp only [kernelMetadata]
  change [_]=[_]
  congr 2
  have choice_eq {A B : Type} (sameType : A=B) (a : Nonempty A) (b : Nonempty B) :
      HEq (Classical.choice a) (Classical.choice b) := by
    cases sameType
    rfl
  apply choice_eq
  exact congrArg (fun t => BackwardData (scalarInterpreter model.mode) model.runtime c upstream t before after) same

/-- The marker simulation is the same actual native execution, artifact,
complete ABI and tape as the numeric simulation. The frame follows from
source preservation and the fixed marker representation; no native write
exclusion or expected byte is supplied. -/
theorem resident_marker_frames {α σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
    (laws : Peepholes (signNumerics engine.native)) (constants : ConstantPorts (signNumerics engine.native))
    (rt : Host.Runtime Marker σ) (versions : σ → Nat → Nat)
    (integer : World Marker σ → Tensor → Nat → Int)
    (groups : CrossEntropy1.Host.Groups (markerObservations integer) rt.toRuntime)
    (objects : CrossEntropy1.Host.ObjectRegistry rt.toRuntime)
    (geometry : CrossEntropy1.Host.ViewGeometry rt.toRuntime)
    (rules : MarkerPrimitiveRules integer rt groups.size groups.rank)
    (o : Options) (logits labels : Tensor) (returned : ForwardReturn)
    (initial saved before after : World α σ) (upstream output work : Tensor) (z : Option Tensor)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) (registered : objects.valid logits)
    (labelDtype : labels.dtype="torch.int64")
    (context : BackwardContext rt.toRuntime objects returned.context.base (markWorld before))
    (upstreamLive : before.live upstream.region)
    (forwardArtifact backwardArtifact : engine.Artifact)
    (forwardABI : CompleteABI (.forward o logits labels))
    (backwardABI : CompleteABI (.backward returned.context saved.native upstream z))
    (forwardTrace backwardTrace : engine.Trace)
    (installedForward : engine.resident _ ((language σ).key _ forwardABI) forwardArtifact)
    (installedBackward : engine.resident _ ((language σ).key _ backwardABI) backwardArtifact)
    (primitivesForward : engine.primitives (markerModel rt versions integer groups.size groups.rank) forwardArtifact)
    (primitivesBackward : engine.primitives (markerModel rt versions integer groups.size groups.rank) backwardArtifact)
    (forwardCoherent : forwardABI.initial=initial.native) (backwardCoherent : backwardABI.initial=before.native)
    (nativeForward : engine.compiler.execute (.forward o logits labels) forwardArtifact forwardABI
      initial.native returned saved.native forwardTrace)
    (nativeBackward : engine.compiler.execute (.backward returned.context saved.native upstream z)
      backwardArtifact backwardABI before.native (backwardTuple output) after.native backwardTrace)
    (workBinding : backwardABI.kernel.outputs=[work])
    (kept : RegionName → Prop)
    (forwardLifetime : Retains engine.layout kept initial.native saved.native)
    (backwardLifetime : Retains engine.layout kept before.native after.native) :
    CallerFrames engine.layout engine.observe kept initial saved before after returned.context.options.inplace work output := by
  let model := markerModel rt versions integer groups.size groups.rank
  let primitives : PrimitiveRules model :=
    ⟨rules.library,rules.arithmetic,rules.zeros,rules.integers,rules.collectives,groups,objects,geometry,rules.allocation⟩
  obtain ⟨markedSaved,forwardSuccess,savedNative,forwardMetadata,forwardRepresentation⟩ :=
    (correct.refinement laws constants).simulate model (.forward o logits labels) forwardArtifact forwardABI
      (markWorld initial) returned saved.native forwardTrace installedForward ⟨primitivesForward,⟨primitives⟩⟩
      forwardCoherent (marked_initial initial) nativeForward
  change markedSaved.native=saved.native at savedNative
  let call : Entry.Invocation (markerObservations integer) markerFP :=
    ⟨rt,rules.library,groups,objects,geometry,o,logits,labels,markWorld initial,markedSaved,returned,
      forwardSuccess,logitsLive,labelsLive,rank,registered⟩
  let aggregation := Classical.choice (forward_aggregation call.data)
  obtain ⟨lossPeers,lsePeers,inputs⟩ := actual_forward_inputs rules.collectives aggregation
  obtain ⟨fs,fl,forwardRep⟩ := forwardRepresentation
  have forwardRep : MarkerRep engine.layout engine.observe initial.native markedSaved := forwardRep
  obtain ⟨markedAfter,backwardSuccess,afterNative,backwardMetadata,backwardRepresentation⟩ :=
    (correct.refinement laws constants).simulate model (.backward returned.context saved.native upstream z)
      backwardArtifact backwardABI (markWorld before) (backwardTuple output) after.native backwardTrace
      installedBackward ⟨primitivesBackward,⟨primitives⟩⟩ backwardCoherent (marked_initial before) nativeBackward
  change markedAfter.native=after.native at afterNative
  obtain ⟨actualOutput,entry,tuple⟩ := python_success markerFP rt versions returned.context saved.native upstream z
    (markWorld before) markedAfter (backwardTuple output) backwardSuccess
  have outputEq : actualOutput=output := by
    simpa only [backwardTuple,List.cons.injEq,Option.some.injEq,and_true] using tuple.symm
  subst actualOutput
  let b := Classical.choice (backward_success markerFP rt returned.context upstream output z (markWorld before) markedAfter
    (backward_entry_success markerFP rt versions returned.context saved.native upstream output z (markWorld before) markedAfter entry).2)
  obtain ⟨kernel⟩ := backward_kernel_data (markerObservations integer) rt rules.library objects b context upstreamLive
  have workEq : b.work=work := by
    have metadata := backward_metadata_work model returned.context saved.native upstream output z (markWorld before) markedAfter entry
    have same : [b.work]=[work] := metadata.symm.trans ((congrArg KernelABI.outputs backwardMetadata).trans workBinding)
    exact List.singleton_inj.mp same
  obtain ⟨bs,bl,backwardRep⟩ := backwardRepresentation
  have backwardRep : MarkerRep engine.layout engine.observe before.native markedAfter := backwardRep
  have forwardLifetime' : Retains engine.layout kept initial.native markedSaved.native := by
    simpa only [savedNative] using forwardLifetime
  have backwardLifetime' : Retains engine.layout kept before.native markedAfter.native := by
    simpa only [afterNative] using backwardLifetime
  constructor
  · intro r held live active offset inside
    have frame := forward_marker_byte engine.layout engine.observe kept integer rt groups.size groups.rank rules
      call.kernel labelDtype call.size aggregation lossPeers lsePeers inputs forwardRep forwardLifetime'
      r held live active offset inside
    simpa only [call,savedNative] using frame
  · intro r held live active offset inside outsideWork outsideOutput
    have frame := backward_marker_byte engine.layout engine.observe kept integer rt rules.library kernel backwardRep
      backwardLifetime' r held live active offset inside (by simpa only [workEq] using outsideWork) outsideOutput
    simpa only [afterNative] using frame
  · intro outOfPlace r held live active offset inside
    have frame := backward_out_of_place_marker_byte engine.layout engine.observe kept integer rt rules.library kernel
      backwardRep backwardLifetime' context.logitsLive outOfPlace r held live active offset inside
    simpa only [afterNative] using frame

#axiomsClean actual_forward_inputs
#axiomsClean marked_context
#axiomsClean backward_metadata_work
#axiomsClean resident_marker_frames
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
