import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SavedOrigins

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Step TensorLibrary IntegerViews CollectiveLibrary SavedPreserves value)
open CrossEntropy1.Physical (Layout Byte byteOf flipSign tensorByte)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

def ReturnedValue {σ : Type} {selection : Selection σ} {fusion}
    (layout : Layout σ) (observe : σ → Nat → Nat → Byte) (reference : Numerics)
    (trace : OriginTrace selection fusion (rawWordAt layout observe))
    (native : σ) (tensor : Tensor) (index : Nat) (expected : SignedExpr) : Prop :=
  ∃ observation : Observation trace tensor index expected, ∀ byte,
    tensorByte layout observe native tensor index byte=
      byteOf (if tensor.isNeg then flipSign (observation.reference reference)
        else observation.reference reference) byte

theorem NumericRep.returnedValue {σ : Type} {selection : Selection σ} {fusion}
    {layout : Layout σ} {observe native trace source exports}
    (rep : NumericRep (selection:=selection) (fusion:=fusion) layout observe native trace source exports)
    (integer : World SignedExpr σ → Tensor → Nat → Int) (tensor : Tensor) (member : tensor∈exports)
    (dtype : tensor.dtype="torch.float32") (index : Nat) (inside : index<tensor.numel)
    (expected : SignedExpr) (logical : value (numericObservations integer) source tensor index=expected)
    (reference : Numerics) (ports : Obligations (signNumerics native) (signNumerics reference)) :
    ReturnedValue layout observe reference trace source.native tensor index expected := by
  let observation := Classical.choice (rep.observation integer tensor member dtype index inside expected logical)
  have exposed := rep.exposed tensor member dtype index inside
  exact ⟨observation,rep.reference_bytes observation exposed.1 exposed.2.1 reference ports⟩

def invocationModel {σ : Type} {integer : World SignedExpr σ → Tensor → Nat → Int}
    (call : Entry.Invocation (numericObservations integer) signedFP) (versions : σ → Nat → Nat) : Model σ :=
  ⟨.numeric,call.runtime,versions,integer,call.groups.size,call.groups.rank⟩

/-- The missing numeric source-to-resident composition. It derives all three
returned output families from the fixed resident simulation and the existing
polymorphic complete source theorem. No NativeBindings, LinkedCode, return
path, output-plan choice or caller-write exclusion is a premise. Frames are
provided separately by the marker composition. -/
theorem complete_resident_values {σ : Type} (engine : Engine σ) (correct : CompilerCorrect engine)
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
    (retained : RegionName → Prop)
    (storage : SavedStorage engine.layout engine.observe retained call.saved before)
    (continuation : OriginContinuation forwardTrace.origins backwardTrace.origins retained)
    (nativeForward : engine.compiler.execute (.forward call.options call.logits call.labels) forwardArtifact forwardABI
      call.initial.native call.returned call.saved.native forwardTrace)
    (nativeBackward : engine.compiler.execute (.backward call.returned.context call.saved.native upstream zUpstream)
      backwardArtifact backwardABI before.native (backwardTuple output) after.native backwardTrace) :
    (∀ row, row<call.data.rows → ReturnedValue engine.layout engine.observe reference forwardTrace.origins
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
      .ok [some output,none,none,none,none,none,none,none] after := by
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
  obtain ⟨backwardSuccess,backwardLabels,backwardRep⟩ := backward.2.2
  have backwardRep : NumericRep engine.layout engine.observe engine.native backwardTrace.origins after
      (exports (.backward call.returned.context call.saved.native upstream zUpstream) (backwardTuple output)) := backwardRep
  let b := call.backwardData versions upstream output zUpstream before after success
  have source := complete_graph (numericObservations integer) signedHost call.runtime call.library ops zeros ints call.objects
    call.groups.size call.groups.rank comms call.kernel call.views dtype labelDtype call.size aggregation lossPeers lsePeers inputs
    between upstream output upstreamLive upstreamShape upstreamDtype b
  have arrays := forward_graph_values (numericObservations integer) signedHost call.runtime call.library ops zeros ints call.objects
    call.groups.size call.groups.rank comms call.kernel call.views labelDtype call.size aggregation lossPeers lsePeers inputs
  have ctx := between.context arrays.2.2.2.context
  obtain ⟨bk⟩ := backward_kernel_data (numericObservations integer) call.runtime call.library call.objects b ctx upstreamLive
  have outputShape : output.shape=call.logits.shape := bk.outputShape.trans arrays.2.2.2.logits.metadata.shape
  have contextDtype : call.returned.context.logits.dtype="torch.float32" := arrays.2.2.2.logits.metadata.dtype.trans dtype
  have outputDtype : output.dtype="torch.float32" := by
    by_cases inplace : call.returned.context.options.inplace=Bool.true
    · have identity : output=call.returned.context.inplaceLogits := by simpa only [if_pos inplace] using b.outputIdentity
      rw [identity,call.data.returnIdentity]
      exact call.views.inplace.metadata.dtype.trans dtype
    · have firstCall : call.runtime.emptyLike call.returned.context.logits Bool.false b.upstreamReady=.ok b.first b.allocated := by
        simpa only [if_neg inplace] using b.firstCall
      have allocated := call.library.tensor.emptyLike _ _ _ _ _
        (bk.snapshot.materialized.frame.live _ ctx.logitsLive) firstCall
      have identity : output=b.first := by simpa only [if_neg inplace] using b.outputIdentity
      exact identity ▸ (allocated.1.dtypeEq.trans contextDtype)
  refine ⟨?_,?_,?_,b.outputIdentity,python⟩
  · intro row hr
    exact forwardRep.returnedValue integer call.returned.losses (by simp [exports]) arrays.1.dtype row
      (by simpa [Tensor.numel,arrays.1.shapeEq] using hr) _ (source.1 row hr).1 reference ports
  · intro row hr
    exact forwardRep.returnedValue integer call.returned.zLosses (by simp [exports]) arrays.2.2.1.dtype row
      (by simpa [Tensor.numel,arrays.2.2.1.shapeEq] using hr) _ (source.1 row hr).2 reference ports
  · intro row hr col hc
    have bound : row*call.data.cols+col<output.numel := by
      simp only [Tensor.numel,outputShape,call.data.logitsShape,List.prod_cons,List.prod_nil,Nat.mul_one]
      have product := Nat.mul_le_mul_right call.data.cols (Nat.succ_le_of_lt hr)
      simp only [Nat.succ_mul] at product
      omega
    exact backwardRep.returnedValue integer output (by simp [exports,backwardTuple]) outputDtype _ bound _
      (source.2 row hr col hc) reference ports

#axiomsClean NumericRep.returnedValue
#axiomsClean complete_resident_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
