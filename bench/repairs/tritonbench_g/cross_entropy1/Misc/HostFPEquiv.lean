import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.EntryFrames
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AutogradReturn
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeEntry
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CompilerSites
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeInputs
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CommunicationProtocol

namespace VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv
open VeriTile.Triton Source Host Physical Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Successful source/host calls on the peers, with group and shape metadata.
No numerical peer result is supplied. -/
structure PeerCalls {σ : Type} (obs : Observations Expr σ) (world rows cols : Nat) where
  calls : Nat → Entry.Invocation obs algebra
  operations : ∀ rank, TensorLibrary obs hostArithmetic (calls rank).runtime
  splits : ∀ rank, (calls rank).data.world>1 ∨ (calls rank).data.cols>65536
  rowsEq : ∀ rank, rank<world → (calls rank).data.rows=rows
  colsEq : ∀ rank, rank<world → (calls rank).data.cols=cols
  worldEq : ∀ rank, rank<world → (calls rank).data.world=world
  rankEq : ∀ rank, rank<world → (calls rank).data.rank=rank

def PeerCalls.peers {σ : Type} {obs : Observations Expr σ} {world rows cols : Nat}
    (group : PeerCalls obs world rows cols) : Nat → Floating.Peer obs :=
  fun rank => (group.calls rank).fpPeer (group.operations rank) (group.splits rank)

inductive Messages {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra) where
  | direct (single : ¬(call.data.world>1 ∨ call.data.cols>65536))
  | split (split : call.data.world>1 ∨ call.data.cols>65536)
      (group : PeerCalls obs call.data.world call.data.rows call.data.cols)

def Messages.loss {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Nat → Expr
  | .direct _ => fun _ => .raw 0
  | .split _ group => Floating.peerLoss group.peers call.data.rows

def Messages.lse {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Nat → Expr
  | .direct _ => fun _ => .raw 0
  | .split _ group => Floating.peerLSE group.peers call.data.rows

def Messages.aggregation {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} :
    Messages call → ForwardAggregation call.data
  | .direct _ => call.aggregation
  | .split flag _ => .split flag (call.splitData flag)

theorem direct_messages {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra)
    (comms : CollectiveLibrary obs hostArithmetic call.runtime call.groups.size call.groups.rank)
    (single : ¬(call.data.world>1 ∨ call.data.cols>65536)) :
    ForwardInputs comms (Messages.aggregation (.direct single)) (Messages.loss (.direct single)) (Messages.lse (.direct single)) := by
  change ForwardInputs comms call.aggregation _ _
  cases call.aggregation with
  | direct => trivial
  | split flag => exact False.elim (single flag)

theorem transmitted_sources {σ : Type} (obs : Observations Expr σ) (world rows cols : Nat)
    (group : PeerCalls obs world rows cols) (rank row : Nat) (rankBound : rank<world) (rowBound : row<rows) :
    Floating.peerLoss group.peers rows (rank*rows+row)=
      localResult hostArithmetic.sum (group.calls rank).data.count rows (group.calls rank).kernel.lossValues row ∧
    Floating.peerLSE group.peers rows (rank*rows+row)=
      localResult hostArithmetic.logsumexp (group.calls rank).data.count rows (group.calls rank).kernel.lseValues row :=
  Floating.transmitted_graphs obs group.peers rows rank row (group.rowsEq rank rankBound) rowBound

/-- Reference returned bytes: exact ordered source/host graph with explicit
local FMA lowering. The native memory encoding and reference numerical ports
remain distinct until the primitive obligations are applied. -/
def ReturnedBytes {σ : Type} (layout : Layout σ) (profile : ReductionProfile) (native reference : Numerics)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) (messages : Messages call)
    (before after : World Expr σ) (upstream output : Tensor)
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after) : Prop :=
  (∀ row, row<call.data.rows → ∀ byte : Fin 4,
    bytes layout (word profile native) call.saved (layout.buffer call.saved.native call.returned.losses.region).device
      ((layout.buffer call.saved.native call.returned.losses.region).base+4*call.returned.losses.index row+byte.val)=
    byteOf (if call.returned.losses.isNeg then flipSign (runInstruction reference (compile profile
      (finalLoss call.kernel hostArithmetic messages.loss messages.lse row)).value)
      else runInstruction reference (compile profile (finalLoss call.kernel hostArithmetic messages.loss messages.lse row)).value) byte) ∧
  (∀ row, row<call.data.rows → ∀ col, col<call.data.cols → ∀ byte : Fin 4,
    bytes layout (word profile native) after (layout.buffer after.native output.region).device
      ((layout.buffer after.native output.region).base+4*output.index (row*call.data.cols+col)+byte.val)=
    byteOf (if output.isNeg then flipSign (runInstruction reference (compile profile
      (gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
        (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
        (value obs before upstream row) col)).value)
      else runInstruction reference (compile profile
      (gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
        (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
        (value obs before upstream row) col)).value) byte)

/-- Complete returned-byte theorem for the explicitly selected scalar graph.
An installed-native claim additionally needs source/object/descriptor binding
of `profile`; this theorem alone does not establish that binding.
It extracts the forward and backward traces,
binds rank inputs to actual peer-local calls, derives the saved LSE and actual
return backing, and transports only primitive numerical ports. It does not
assert that the installed GPU implements those ports. -/
theorem symbolic_complete {σ : Type} (layout : Layout σ) (profile : ReductionProfile) (native reference : Numerics)
    (ports : Obligations native reference) (kept : RegionName → Prop)
    (obs : Observations Expr σ) (negation : EncodesNegation (word profile native) obs)
    (call : Entry.Invocation obs algebra) (ops : TensorLibrary obs hostArithmetic call.runtime) (ints : IntegerViews obs)
    (life : LifetimeLibrary layout call.runtime kept) (backing : BackingLibrary layout call.runtime)
    (comms : CollectiveLibrary obs hostArithmetic call.runtime call.groups.size call.groups.rank)
    (dtype : call.logits.dtype="torch.float32") (labelDtype : call.labels.dtype="torch.int64")
    (messages : Messages call) (binding : ForwardInputs comms messages.aggregation messages.loss messages.lse)
    (versions : σ → Nat → Nat) (before after : World Expr σ)
    (between : SavedPreserves obs call.returned.context call.saved before) (betweenNative : Retains layout kept call.saved.native before.native)
    (upstream output : Tensor) (upstreamLive : before.live upstream.region)
    (upstreamShape : upstream.shape=[call.data.rows]) (upstreamDtype : upstream.dtype="torch.float32")
    (success : Autograd.backwardEntry algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after)
    (heldLoss : kept call.returned.losses.region) (heldLogits : kept call.returned.context.logits.region)
    (heldWork : kept (call.backwardData versions upstream output before after success).work.region) :
    ReturnedBytes layout profile native reference obs call messages before after upstream output
      (call.backwardData versions upstream output before after success) ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output before after success).first) ∧
    CallerFrames layout (word profile native) kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output before after success).work output ∧
    Autograd.pythonBackward algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=
      .ok [some output,none,none,none,none,none,none,none] after := by
  have result := complete_returned_bytes layout profile native kept obs negation call.runtime call.library ops ints call.objects life backing
    call.groups.size call.groups.rank comms call.kernel call.views dtype labelDtype call.size messages.aggregation
    messages.loss messages.lse binding between betweenNative upstream output upstreamLive upstreamShape upstreamDtype
    (call.backwardData versions upstream output before after success) heldLoss heldLogits heldWork
  refine ⟨?_,(call.backwardData versions upstream output before after success).outputIdentity,?_,
    Autograd.python_return algebra call.runtime versions _ upstream output before after success⟩
  · rcases result with ⟨lossBytes,gradientBytes⟩
    constructor
    · intro row hr byte
      have value := lossBytes row hr byte
      simpa only [compiled_word,evaluation_equiv profile native reference ports] using value
    · intro row hr col hc byte
      have value := gradientBytes row hr col hc byte
      simpa only [compiled_word,evaluation_equiv profile native reference ports] using value
  · exact entry_frames layout (word profile native) kept obs algebra hostArithmetic call ops ints life comms labelDtype
      messages.aggregation messages.loss messages.lse binding versions before after between upstream output upstreamLive success

/-- The admitted runtime is this actual invocation or one of its actual peers. -/
def Messages.allows {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (messages : Messages call) (rt : Runtime Expr σ) : Prop :=
  rt=call.runtime ∨ match messages with
    | .direct _ => False
    | .split _ group => ∃ rank, rank<call.data.world ∧ rt=(group.calls rank).runtime

def Messages.inputViews {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (messages : Messages call) (layout : Layout σ) (observe : σ → Nat → Nat → Physical.Byte) : Prop :=
  match messages with
  | .direct _ => True
  | .split _ group => ∀ rank, rank<call.data.world → InitialInputViews layout observe obs (group.calls rank)

/-- One collective invocation includes this very local rank and the same
process group on each actual peer. No numerical peer output is supplied. -/
def Messages.Compatible {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Prop
  | .direct _ => True
  | .split _ group => group.calls call.data.rank=call ∧
      ∀ rank, rank < call.data.world → (group.calls rank).options.processGroup=call.options.processGroup

def Messages.expectedRank {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (_messages : Messages call) (location : SourceLocation) : Nat :=
  match location.participant with
  | .main | .backward => call.data.rank
  | .peer rank => rank

/-- Exact main/peer/backward source universe. The local peer aliases main;
it is not a second independently numbered physical invocation. -/
def Messages.source {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (messages : Messages call) (abi : InvocationABI σ) {upstream output before after}
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after)
    (location : SourceLocation) (record : SourceCall σ) : Prop :=
  messages.Compatible ∧ abi.rank location=messages.expectedRank location ∧
  (∀ rank, abi.peerParticipant rank=if rank=call.data.rank then .main else .peer rank) ∧
  match location.participant with
  | .main => record.runtime=call.runtime ∧ ForwardMember call location.slot record
  | .backward => record.runtime=call.runtime ∧ BackwardMember b location.slot record
  | .peer rank => match messages with
    | .direct _ => False
    | .split _ group => rank < call.data.world ∧ rank≠call.data.rank ∧
        record.runtime=(group.calls rank).runtime ∧ ForwardMember (group.calls rank) location.slot record

structure LedgerBoundaries {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    {messages : Messages call} {abi upstream output before after}
    {b : BackwardData algebra call.runtime call.returned.context upstream output before after}
    (ledger : HostLedger (messages.source abi b)) : Prop where
  forwardInitial : ledger.state .main 0=call.initial.native
  forwardFinal : ledger.state .main (ledger.entries .main).length=call.saved.native
  backwardInitial : ledger.state .backward 0=before.native
  backwardFinal : ledger.state .backward (ledger.entries .backward).length=after.native

/-- Source-linked actual execution data. Values are never premises: the code
certificate is made of actual primitive events, and the write certificate is
address-only. Each source occurrence selects its own full call descriptor. -/
structure NativeEvidence {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Physical.Byte)
    (events : ByteEvents σ) (selection : Selection σ) (trace : PrimitiveTrace σ)
    (readWord : σ → Tensor → Nat → Floating.Word)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) (messages : Messages call)
    (before after : World Expr σ) (upstream output : Tensor)
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after)
    (kept : RegionName → Prop) where
  loss : ∀ row, row<call.data.rows → SelectedReturn (selection:=selection) layout observe events trace readWord
    (finalLoss call.kernel hostArithmetic messages.loss messages.lse row) call.saved.native call.returned.losses row
  gradient : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
    SelectedReturn (selection:=selection) layout observe events trace readWord
      (gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
        (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
        (value obs before upstream row) col) after.native output (row*call.data.cols+col)
  footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
    call.returned.context.options.inplace b.work output

/-- Structural selected-object bindings. Executions are derived below from
one-instruction platform interfaces, rather than assumed at task outputs. -/
structure NativeBindings {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Physical.Byte)
    (events : ByteEvents σ) (selection : Selection σ) (registry : InstructionRegistry σ)
    (readWord : σ → Tensor → Nat → Floating.Word)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) (messages : Messages call)
    (before after : World Expr σ) (upstream output : Tensor)
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after)
    (kept : RegionName → Prop) where
  loss : ∀ row, row<call.data.rows → LinkedReturn (selection:=selection) layout observe events registry readWord
    (finalLoss call.kernel hostArithmetic messages.loss messages.lse row) call.saved.native call.returned.losses row
  gradient : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
    LinkedReturn (selection:=selection) layout observe events registry readWord
      (gradientScalar algebra b.parameters (obs.integer call.initial call.labels row)
        (value obs call.initial call.logits (row*call.data.cols+col)) (finalLSE call.kernel hostArithmetic messages.lse row)
        (value obs before upstream row) col) after.native output (row*call.data.cols+col)
  footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
    call.returned.context.options.inplace b.work output

def NativeBindings.executed {σ : Type} {layout : Layout σ} {observe events selection registry trace readWord}
    {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} {messages : Messages call}
    {before after : World Expr σ} {upstream output : Tensor}
    {b : BackwardData algebra call.runtime call.returned.context upstream output before after} {kept}
    (binding : NativeBindings layout observe events selection registry readWord obs call messages before after upstream output b kept)
    (api : InstructionExecution registry trace) :
    NativeEvidence layout observe events selection trace readWord obs call messages before after upstream output b kept :=
  ⟨fun row hr => (binding.loss row hr).executed api,
    fun row hr col hc => (binding.gradient row hr col hc).executed api,binding.footprints⟩

def NativeReturnedBytes {σ : Type} {layout : Layout σ} {observe events selection trace readWord}
    {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} {messages : Messages call}
    {before after : World Expr σ} {upstream output : Tensor}
    {b : BackwardData algebra call.runtime call.returned.context upstream output before after} {kept}
    (evidence : NativeEvidence layout observe events selection trace readWord obs call messages before after upstream output b kept)
    (reference : Numerics) : Prop :=
  (∀ row (hr : row<call.data.rows) (byte : Fin 4),
    tensorByte layout observe call.saved.native call.returned.losses row byte=
      byteOf (if call.returned.losses.isNeg then flipSign ((evidence.loss row hr).plan.reference reference)
        else (evidence.loss row hr).plan.reference reference) byte) ∧
  (∀ row (hr : row<call.data.rows) col (hc : col<call.data.cols) (byte : Fin 4),
    tensorByte layout observe after.native output (row*call.data.cols+col) byte=
      byteOf (if output.isNeg then flipSign ((evidence.gradient row hr col hc).plan.reference reference)
        else (evidence.gradient row hr col hc).plan.reference reference) byte)

/-- Installed execution now uses the registry projected from the actual
loaded decoder image. Exact source slots, operand order, request identity and
continuous memory paths constrain the same events used for the byte proof.
The compiler/runtime/NCCL interfaces are explicit local structural contracts;
all numerical deferrals remain fixed-arity scalar ports. -/
theorem complete {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Physical.Byte)
    (events : ByteEvents σ) (selection : Selection σ) (trace : PrimitiveTrace σ)
    (native reference : Numerics)
    (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native)
    (api : PrimitiveContracts trace native (logicalWord layout observe) (tensorByte layout observe))
    (memory : ByteContracts events observe selection)
    (kept : RegionName → Prop) (obs : Observations Expr σ) (call : Entry.Invocation obs algebra)
    (inputViews : InitialInputViews layout observe obs call)
    (messages : Messages call) (peerViews : messages.inputViews layout observe)
    (versions : σ → Nat → Nat) (before after : World Expr σ) (upstream output : Tensor)
    (upstreamViews : UpstreamInputViews layout observe obs before upstream call.data.rows)
    (success : Autograd.backwardEntry algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after)
    (abi : InvocationABI σ) (loader : InstalledLoader σ)
    (objects : ObjectSelection (messages.source abi (call.backwardData versions upstream output before after success)) abi selection loader.loaded)
    (traceImage : TraceImage objects.registry trace)
    (localMemory : ImageMemory loader.loaded observe)
    (ledger : HostLedger (messages.source abi (call.backwardData versions upstream output before after success)))
    (boundaries : LedgerBoundaries ledger)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (machine : LoaderContract loader.loaded trace (StepPermit objects ledger layout events trace order requests))
    (sourceOf : String → String)
    (sourcePins : ∀ descriptor invoked,
      ((objects.exact descriptor invoked).call.kind=.forward ∨ (objects.exact descriptor invoked).call.kind=.backward) →
      sourceOf descriptor.objectDigest=repairedSourceSHA256)
    (binding : NativeBindings layout observe events selection objects.registry (logicalWord layout observe) obs call messages before after upstream output
      (call.backwardData versions upstream output before after success) kept)
    (lossPaths : ∀ row (hr : row < call.data.rows), ReturnPath objects ledger .main (binding.loss row hr))
    (gradientPaths : ∀ row (hr : row < call.data.rows) col (hc : col < call.data.cols),
      ReturnPath objects ledger .backward (binding.gradient row hr col hc))
    (forwardSteps : LedgerSteps objects ledger events .main 0 (ledger.entries .main).length binding.footprints.forwardWrites)
    (backwardSteps : LedgerSteps objects ledger events .backward 0 (ledger.entries .backward).length binding.footprints.backwardWrites) :
    let instructions := objects.instructions machine protocol.permits
    let evidence := binding.executed instructions
    NativeReturnedBytes evidence reference ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output before after success).first) ∧
    NativeCallerFrames layout observe kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output before after success).work output ∧
    Autograd.pythonBackward algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=
      .ok [some output,none,none,none,none,none,none,none] after ∧
    (∀ s op out, Emitted trace s out op →
      Nonempty (ExactObject (messages.source abi (call.backwardData versions upstream output before after success)) abi loader.loaded s.call (objects.image s.call)) ∧
      ∀ value∈op.operands, objects.registry.active (order.owner value) ∧
        objects.registry.output (order.owner value)=value ∧
        ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects s)) ∧
    (∀ d (called : selection.invoked d), (objects.exact d called).call.kind=.reduceWait →
      ∃ work : CollectiveWork objects requests, work.wait=d) ∧
    (∀ s channel sequence sourceRank destinationRank value out,
      trace.transfer s (.rank channel sequence sourceRank destinationRank) value out →
      Nonempty (RankRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank) ∨
        Nonempty (GatherRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank)) ∧
    (∀ descriptor invoked,
      ((objects.exact descriptor invoked).call.kind=.forward ∨ (objects.exact descriptor invoked).call.kind=.backward) →
      sourceOf descriptor.objectDigest=repairedSourceSHA256) ∧
    InitialInputViews layout observe obs call ∧ messages.inputViews layout observe ∧
    UpstreamInputViews layout observe obs before upstream call.data.rows := by
  let instructions := objects.instructions machine protocol.permits
  let evidence := binding.executed instructions
  let footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
      call.returned.context.options.inplace (call.backwardData versions upstream output before after success).work output :=
    { binding.footprints with
      forwardTrace := by
        have steps := forwardSteps.trace
        simpa only [boundaries.forwardInitial,boundaries.forwardFinal] using steps
      backwardTrace := by
        have steps := backwardSteps.trace
        simpa only [boundaries.backwardInitial,boundaries.backwardFinal] using steps }
  change NativeReturnedBytes evidence reference ∧ _
  refine ⟨?_,(call.backwardData versions upstream output before after success).outputIdentity,
    native_caller_frames memory footprints,
    Autograd.python_return algebra call.runtime versions _ upstream output before after success,?_,protocol.waits,?_,
    sourcePins,inputViews,peerViews,upstreamViews⟩
  · constructor
    · intro row hr byte
      exact (lossPaths row hr).completion.bytes instructions api localMemory memory ports laws constants byte
    · intro row hr col hc byte
      exact (gradientPaths row hr col hc).completion.bytes instructions api localMemory memory ports laws constants byte
  · intro s op out emitted
    have h := traceImage.event s op out emitted
    exact ⟨⟨objects.exact s.call h.1.1⟩,protocol.operands s op h.1 h.2.1⟩
  · intro s channel sequence sourceRank destinationRank value out transfer
    have h := traceImage.event s (.transfer (.rank channel sequence sourceRank destinationRank) value) out transfer
    exact protocol.ranks s channel sequence sourceRank destinationRank value h.1 h.2.1

theorem checked_complete {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Physical.Byte)
    (events : ByteEvents σ) (selection : Selection σ) (trace : PrimitiveTrace σ)
    (native reference : Numerics)
    (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native)
    (api : PrimitiveContracts trace native (logicalWord layout observe) (tensorByte layout observe))
    (memory : ByteContracts events observe selection)
    (kept : RegionName → Prop) (obs : Observations Expr σ) (call : Entry.Invocation obs algebra)
    (metadata : Checked.ForwardMetadata)
    (admission : Checked.InputContract metadata call.options call.logits call.labels (obs.integer call.initial call.labels))
    (inputViews : InitialInputViews layout observe obs call)
    (messages : Messages call) (peerViews : messages.inputViews layout observe)
    (versions : σ → Nat → Nat) (before after : World Expr σ) (upstream output : Tensor)
    (upstreamMetadata : Checked.BackwardMetadata)
    (upstreamAdmission : Checked.BackwardContract upstreamMetadata call.data.rows call.returned.context.logits upstream)
    (upstreamViews : UpstreamInputViews layout observe obs before upstream call.data.rows)
    (success : Autograd.backwardEntry algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after)
    (abi : InvocationABI σ) (loader : InstalledLoader σ)
    (objects : ObjectSelection (messages.source abi (call.backwardData versions upstream output before after success)) abi selection loader.loaded)
    (traceImage : TraceImage objects.registry trace)
    (localMemory : ImageMemory loader.loaded observe)
    (ledger : HostLedger (messages.source abi (call.backwardData versions upstream output before after success)))
    (boundaries : LedgerBoundaries ledger)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (machine : LoaderContract loader.loaded trace (StepPermit objects ledger layout events trace order requests))
    (sourceOf : String → String)
    (sourcePins : ∀ descriptor invoked,
      ((objects.exact descriptor invoked).call.kind=.forward ∨ (objects.exact descriptor invoked).call.kind=.backward) →
      sourceOf descriptor.objectDigest=Checked.sourceSHA256)
    (binding : NativeBindings layout observe events selection objects.registry (logicalWord layout observe) obs call messages before after upstream output
      (call.backwardData versions upstream output before after success) kept)
    (lossPaths : ∀ row (hr : row < call.data.rows), ReturnPath objects ledger .main (binding.loss row hr))
    (gradientPaths : ∀ row (hr : row < call.data.rows) col (hc : col < call.data.cols),
      ReturnPath objects ledger .backward (binding.gradient row hr col hc))
    (forwardSteps : LedgerSteps objects ledger events .main 0 (ledger.entries .main).length binding.footprints.forwardWrites)
    (backwardSteps : LedgerSteps objects ledger events .backward 0 (ledger.entries .backward).length binding.footprints.backwardWrites) :
    let instructions := objects.instructions machine protocol.permits
    let evidence := binding.executed instructions
    Checked.forward algebra call.runtime metadata call.options call.logits call.labels
      (obs.integer call.initial call.labels) call.initial=.ok call.returned call.saved ∧
    Checked.backward algebra call.runtime versions (Autograd.capture versions call.returned.context call.saved.native)
      upstreamMetadata call.data.rows upstream before=.ok output after ∧
    NativeReturnedBytes evidence reference ∧
    output=(if call.returned.context.options.inplace then call.returned.context.inplaceLogits
      else (call.backwardData versions upstream output before after success).first) ∧
    NativeCallerFrames layout observe kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output before after success).work output ∧
    Autograd.pythonBackward algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=
      .ok [some output,none,none,none,none,none,none,none] after ∧
    (∀ s op out, Emitted trace s out op →
      Nonempty (ExactObject (messages.source abi (call.backwardData versions upstream output before after success)) abi loader.loaded s.call (objects.image s.call)) ∧
      ∀ value∈op.operands, objects.registry.active (order.owner value) ∧
        objects.registry.output (order.owner value)=value ∧
        ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects s)) ∧
    (∀ d (called : selection.invoked d), (objects.exact d called).call.kind=.reduceWait →
      ∃ work : CollectiveWork objects requests, work.wait=d) ∧
    (∀ s channel sequence sourceRank destinationRank value out,
      trace.transfer s (.rank channel sequence sourceRank destinationRank) value out →
      Nonempty (RankRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank) ∨
        Nonempty (GatherRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank)) ∧
    (∀ descriptor invoked,
      ((objects.exact descriptor invoked).call.kind=.forward ∨ (objects.exact descriptor invoked).call.kind=.backward) →
      sourceOf descriptor.objectDigest=Checked.sourceSHA256) ∧
    InitialInputViews layout observe obs call ∧ messages.inputViews layout observe ∧
    UpstreamInputViews layout observe obs before upstream call.data.rows := by
  let instructions := objects.instructions machine protocol.permits
  let evidence := binding.executed instructions
  let footprints : CallerWriteFootprints events layout kept call.initial call.saved before after
      call.returned.context.options.inplace (call.backwardData versions upstream output before after success).work output :=
    { binding.footprints with
      forwardTrace := by
        have steps := forwardSteps.trace
        simpa only [boundaries.forwardInitial,boundaries.forwardFinal] using steps
      backwardTrace := by
        have steps := backwardSteps.trace
        simpa only [boundaries.backwardInitial,boundaries.backwardFinal] using steps }
  refine ⟨Checked.source_success call metadata admission,
    Checked.backward_source_success call versions before after upstream output upstreamMetadata upstreamAdmission success,?_⟩
  change NativeReturnedBytes evidence reference ∧ _
  refine ⟨?_,(call.backwardData versions upstream output before after success).outputIdentity,
    native_caller_frames memory footprints,
    Autograd.python_return algebra call.runtime versions _ upstream output before after success,?_,protocol.waits,?_,
    sourcePins,inputViews,peerViews,upstreamViews⟩
  · constructor
    · intro row hr byte
      exact (lossPaths row hr).completion.bytes instructions api localMemory memory ports laws constants byte
    · intro row hr col hc byte
      exact (gradientPaths row hr col hc).completion.bytes instructions api localMemory memory ports laws constants byte
  · intro s op out emitted
    have h := traceImage.event s op out emitted
    exact ⟨⟨objects.exact s.call h.1.1⟩,protocol.operands s op h.1 h.2.1⟩
  · intro s channel sequence sourceRank destinationRank value out transfer
    have h := traceImage.event s (.transfer (.rank channel sequence sourceRank destinationRank) value) out transfer
    exact protocol.ranks s channel sequence sourceRank destinationRank value h.1 h.2.1


#axiomsClean checked_complete
#axiomsClean NativeBindings.executed
#axiomsClean complete

/-- The exact local instruction changes used by the pinned compiler sites.
These conclusions compose primitive all-pattern ports, never a whole-output
or whole-expression equivalence assumption. -/
theorem compiler_sites (profile : ReductionProfile) (native reference : Numerics) (ports : Obligations native reference)
    (laws : Peepholes native) (one : ConstantOne native) (exp2 : Floating.Word → Floating.Word)
    (exponential : ExpImplementation native exp2) (scale lse probability smoothing target loss x : Floating.Word) (commuted : Bool) :
    (word profile reference (.binary .add (.raw probability)
      (.binary .mul (.binary .mul (.binary .mul (.literal 2) (.raw scale)) (.raw lse)) (.raw probability)))=
      if commuted then native.fma probability (native.mul (native.add scale scale) lse) probability
      else native.fma (native.mul (native.add scale scale) lse) probability probability) ∧
    (word profile reference (.binary .add (.raw loss) (.binary .mul (.binary .mul (.raw scale) (.raw lse)) (.raw lse)))=
      if commuted then native.fma lse (native.mul scale lse) loss else native.fma (native.mul scale lse) lse loss) ∧
    (word profile reference (.binary .sub (.raw loss) (.binary .mul (.binary .sub (.natural 1) (.raw smoothing)) (.raw target)))=
      native.fma (native.neg (native.sub (native.literal 1) smoothing)) target loss) ∧
    word profile reference (.binary .sub (.raw x) (.literal 1))=native.add x (native.literal (-1)) ∧
    word profile reference (.unary .exp (.raw x))=exp2 (native.mul x 0x3fb8aa3b) := by
  simp only [←evaluation_equiv profile native reference ports]
  exact ⟨backward_correction_site profile native laws scale lse probability commuted,
    squared_lse_site profile native laws scale lse loss commuted,smoothed_target_site profile native one smoothing target loss,
    subtract_one_site profile native laws x,exp_site profile native exp2 exponential x⟩

#axiomsClean symbolic_complete
#axiomsClean transmitted_sources
#axiomsClean compiler_sites
#specNonCircular Obligations avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular Peepholes avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv
