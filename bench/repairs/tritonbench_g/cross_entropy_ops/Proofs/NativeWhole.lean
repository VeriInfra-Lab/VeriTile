import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativePeers
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeRequests
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeProtocol

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Step bind)
open CrossEntropy1.Source (FP32)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def pythonBackward {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (savedNative : σ)
    (upstream : Tensor) (zUpstream : Option Tensor) (before : World α σ) : Step (World α σ) (List (Option Tensor)) :=
  bind (backwardEntry M rt versions c savedNative upstream zUpstream before)
    (fun output after => .ok (backwardTuple output) after)

theorem python_return {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (versions : σ → Nat → Nat) (c : Context) (savedNative : σ)
    (upstream output : Tensor) (zUpstream : Option Tensor) (before after : World α σ)
    (success : backwardEntry M rt versions c savedNative upstream zUpstream before=.ok output after) :
    pythonBackward M rt versions c savedNative upstream zUpstream before=
      .ok [some output,none,none,none,none,none,none,none] after := by
  simp only [pythonBackward,success,CrossEntropy1.Host.bind,backwardTuple]

#axiomsClean python_return
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World Observations TensorLibrary CollectiveLibrary IntegerViews SavedPreserves value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Source-indexed decoded programs and last stores, covering both returned
forward tensors and every gradient coordinate. These are structural code and
address certificates; no expected output word is supplied. -/
structure NativeBindings {σ : Type} (layout : Layout σ) (events : ByteEvents σ) (selection : Selection σ)
    (contract : Occurrence σ → Bool) (registry : InstructionRegistry σ)
    (readWord : σ → Tensor → Nat → CrossEntropy1.Floating.Word)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) (messages : Messages call)
    {before after : World Expr σ} {upstream output : Tensor}
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after)
    (kept : RegionName → Prop) where
  loss : ∀ row, row<call.data.rows → LinkedReturn (selection:=selection) contract registry readWord
    (finalLoss call.kernel hostArithmetic messages.loss messages.lse row) call.returned.losses row
  zloss : ∀ row, row<call.data.rows → LinkedReturn (selection:=selection) contract registry readWord
    (finalZ call.kernel hostArithmetic messages.lse row) call.returned.zLosses row
  gradient : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols → LinkedReturn (selection:=selection) contract registry readWord
    (Source.gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
      (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
      (value obs before upstream row) col) output (row*call.data.cols+col)
  footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
    call.returned.context.options.inplace b.work output

def NativeReturnedBytes {σ : Type} {layout : Layout σ} {events selection contract registry readWord}
    {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} {messages : Messages call}
    {before after : World Expr σ} {upstream output : Tensor}
    {b : BackwardData algebra call.runtime call.returned.context upstream output before after} {kept}
    (binding : NativeBindings layout events selection contract registry readWord obs call messages b kept)
    (observe : σ → Nat → Nat → Byte) (reference : Numerics) : Prop :=
  (∀ row (hr : row<call.data.rows) (byte : Fin 4),
    tensorByte layout observe call.saved.native call.returned.losses row byte=
      byteOf (if call.returned.losses.isNeg then flipSign (ConfiguredPlan.reference contract (binding.loss row hr).plan reference)
        else ConfiguredPlan.reference contract (binding.loss row hr).plan reference) byte) ∧
  (∀ row (hr : row<call.data.rows) (byte : Fin 4),
    tensorByte layout observe call.saved.native call.returned.zLosses row byte=
      byteOf (if call.returned.zLosses.isNeg then flipSign (ConfiguredPlan.reference contract (binding.zloss row hr).plan reference)
        else ConfiguredPlan.reference contract (binding.zloss row hr).plan reference) byte) ∧
  (∀ row (hr : row<call.data.rows) col (hc : col<call.data.cols) (byte : Fin 4),
    tensorByte layout observe after.native output (row*call.data.cols+col) byte=
      byteOf (if output.isNeg then flipSign (ConfiguredPlan.reference contract (binding.gradient row hr col hc).plan reference)
        else ConfiguredPlan.reference contract (binding.gradient row hr col hc).plan reference) byte)

def SourceReturned {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra) (messages : Messages call)
    {before after : World Expr σ} {upstream output : Tensor}
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after) : Prop :=
  (∀ row, row<call.data.rows → value obs call.saved call.returned.losses row=finalLoss call.kernel hostArithmetic messages.loss messages.lse row ∧
    value obs call.saved call.returned.zLosses row=finalZ call.kernel hostArithmetic messages.lse row) ∧
  (∀ row, row<call.data.rows → ∀ col, col<call.data.cols → value obs after output (row*call.data.cols+col)=
    Source.gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
      (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
      (value obs before upstream row) col)

/-- Whole actual invocation: source graph, exact saved context, actual loaded
instructions, ordered operands/communication, last store, native return bytes,
physical sign handling and caller frame. Only fixed-arity scalar numerical
ports remain; the source/decoder/runtime contracts are explicit and separate. -/
theorem complete {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (events : ByteEvents σ) (selection : Selection σ) (trace : PrimitiveTrace σ)
    (native reference : Numerics) (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native)
    (api : PrimitiveContracts trace native (logicalWord layout observe) (tensorByte layout observe))
    (memory : ByteContracts events observe selection) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) (messages : Messages call)
    (ops : TensorLibrary obs hostArithmetic call.runtime.toRuntime) (zeros : ZeroAPI obs call.runtime hostArithmetic.zero)
    (ints : IntegerViews obs) (comms : CollectiveLibrary obs hostArithmetic call.runtime.toRuntime call.groups.size call.groups.rank)
    (dtype : call.logits.dtype="torch.float32") (labelDtype : call.labels.dtype="torch.int64")
    (inputs : ForwardInputs comms messages.aggregation messages.loss messages.lse)
    (inputViews : InitialInputViews layout observe obs call) (peerViews : messages.inputViews layout observe)
    (versions : σ → Nat → Nat) (upstream output : Tensor) (zUpstream : Option Tensor) (before after : World Expr σ)
    (between : SavedPreserves obs call.returned.context.base call.saved before)
    (upstreamLive : before.live upstream.region) (upstreamShape : upstream.shape=[call.data.rows])
    (upstreamDtype : upstream.dtype="torch.float32")
    (upstreamViews : UpstreamInputViews layout observe obs before upstream call.data.rows)
    (success : backwardEntry algebra call.runtime versions call.returned.context call.saved.native upstream zUpstream before=.ok output after)
    (abi : InvocationABI σ) (loader : InstalledLoader σ)
    (objects : ObjectSelection (messages.source abi (call.backwardData versions upstream output zUpstream before after success)) abi selection (loader.loaded abi))
    (traceImage : TraceImage objects.registry trace) (localMemory : ImageMemory (loader.loaded abi) observe)
    (ledger : HostLedger (messages.source abi (call.backwardData versions upstream output zUpstream before after success)))
    (boundaries : LedgerBoundaries ledger) (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (machine : LoaderContract (loader.loaded abi) trace (StepPermit objects ledger layout events trace order requests))
    (allocation : σ → RegionName → Nat) (encode : ℝ → CrossEntropy1.Floating.Word)
    (fusion : ConfiguredPlan.FusionSelection σ) (policy : ConfiguredPlan.Policy fusion)
    (requestBinding : RequestBinding objects layout allocation encode fusion policy)
    (sourceOf : String → String)
    (sourcePins : ∀ d called, ((objects.exact d called).call.kind=.forward ∨ (objects.exact d called).call.kind=.backward) →
      sourceOf d.objectDigest="b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5")
    (binding : NativeBindings layout events selection policy.contract objects.registry (logicalWord layout observe) obs call messages
      (call.backwardData versions upstream output zUpstream before after success) kept)
    (lossPaths : ∀ row (hr : row<call.data.rows), ReturnPath (layout:=layout) (events:=events) objects ledger .main (binding.loss row hr) call.saved.native)
    (zlossPaths : ∀ row (hr : row<call.data.rows), ReturnPath (layout:=layout) (events:=events) objects ledger .main (binding.zloss row hr) call.saved.native)
    (gradientPaths : ∀ row (hr : row<call.data.rows) col (hc : col<call.data.cols),
      ReturnPath (layout:=layout) (events:=events) objects ledger .backward (binding.gradient row hr col hc) after.native)
    (forwardSteps : LedgerSteps objects ledger events .main 0 (ledger.entries .main).length binding.footprints.forwardWrites)
    (backwardSteps : LedgerSteps objects ledger events .backward 0 (ledger.entries .backward).length binding.footprints.backwardWrites) :
    NativeReturnedBytes binding observe reference ∧
    SourceReturned call messages (call.backwardData versions upstream output zUpstream before after success) ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output zUpstream before after success).first) ∧
    NativeCallerFrames layout observe kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output zUpstream before after success).work output ∧
    pythonBackward algebra call.runtime versions call.returned.context call.saved.native upstream zUpstream before=
      .ok [some output,none,none,none,none,none,none,none] after ∧
    (∀ s op out, Emitted trace s out op → ∀ value∈op.operands,
      objects.registry.active (order.owner value) ∧ objects.registry.output (order.owner value)=value ∧
      ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects s)) ∧
    RequestBinding objects layout allocation encode fusion policy ∧
    (∀ d called, ((objects.exact d called).call.kind=.forward ∨ (objects.exact d called).call.kind=.backward) →
      sourceOf d.objectDigest="b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5") ∧
    InitialInputViews layout observe obs call ∧ messages.inputViews layout observe ∧
    UpstreamInputViews layout observe obs before upstream call.data.rows := by
  let instructions := objects.instructions machine protocol.permits
  let footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
      call.returned.context.options.inplace (call.backwardData versions upstream output zUpstream before after success).work output :=
    {binding.footprints with
      forwardTrace := by simpa only [boundaries.forwardInitial,boundaries.forwardFinal] using forwardSteps.trace
      backwardTrace := by simpa only [boundaries.backwardInitial,boundaries.backwardFinal] using backwardSteps.trace}
  refine ⟨?_,?_,(call.backwardData versions upstream output zUpstream before after success).outputIdentity,
    native_caller_frames memory footprints,python_return algebra call.runtime versions _ _ upstream output zUpstream before after success,
    ?_,requestBinding,sourcePins,inputViews,peerViews,upstreamViews⟩
  · refine ⟨?_,?_,?_⟩
    · intro row hr byte
      exact (lossPaths row hr).completion.bytes instructions api localMemory memory ports laws constants byte
    · intro row hr byte
      exact (zlossPaths row hr).completion.bytes instructions api localMemory memory ports laws constants byte
    · intro row hr col hc byte
      exact (gradientPaths row hr col hc).completion.bytes instructions api localMemory memory ports laws constants byte
  · exact complete_graph obs hostArithmetic call.runtime call.library ops zeros ints call.objects call.groups.size call.groups.rank comms
      call.kernel call.views dtype labelDtype call.size messages.aggregation messages.loss messages.lse inputs between upstream output
      upstreamLive upstreamShape upstreamDtype (call.backwardData versions upstream output zUpstream before after success)
  · intro site op out emitted
    have event := traceImage.event site op out emitted
    exact protocol.operands site op event.1 event.2.1

#axiomsClean complete
#specNonCircular Obligations avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular Peepholes avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
