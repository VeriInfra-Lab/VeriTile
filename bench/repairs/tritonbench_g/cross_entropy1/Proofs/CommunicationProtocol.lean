import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CausalExecution

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Triton Source Host Physical
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Exact start/LSE/wait objects from the same participant and source slots.
The arguments copied from RankTrace fix every intervening state and buffer.
Pending evidence is required at BOTH pending and reduced, not merely the
existence of some start with the same numeric handle. -/
structure CollectiveWork {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (requests : RequestRegistry σ) where
  participant : Participant
  start : CallDescriptor σ
  lse : CallDescriptor σ
  wait : CallDescriptor σ
  startCalled : selection.invoked start
  lseCalled : selection.invoked lse
  waitCalled : selection.invoked wait
  startLocation : (objects.exact start startCalled).location=⟨participant,[9,1,2]⟩
  lseLocation : (objects.exact lse lseCalled).location=⟨participant,[9,1,3]⟩
  waitLocation : (objects.exact wait waitCalled).location=⟨participant,[9,1,4]⟩
  dst : Tensor
  gathered : Tensor
  globalLSE : Tensor
  handle : Nat
  group : Nat
  collected : World Expr σ
  pending : World Expr σ
  reduced : World Expr σ
  waited : World Expr σ
  startKind : (objects.exact start startCalled).call.kind=.reduceStart
  lseKind : (objects.exact lse lseCalled).call.kind=.torchLSE
  waitKind : (objects.exact wait waitCalled).call.kind=.reduceWait
  startArguments : (objects.exact start startCalled).call.arguments=
    .collective dst gathered globalLSE handle group collected pending reduced waited
  waitArguments : (objects.exact wait waitCalled).call.arguments=
    .collective dst gathered globalLSE handle group collected pending reduced waited
  startBefore : start.before=collected.native
  startAfter : start.after=pending.native
  lseBefore : lse.before=pending.native
  lseAfter : lse.after=reduced.native
  lseInput : lse.inputs=[gathered]
  lseOutput : lse.outputs=[globalLSE]
  waitBefore : wait.before=reduced.native
  waitAfter : wait.after=waited.native
  startGroup : start.group=group
  waitGroup : wait.group=group
  startHandle : start.work=some handle
  waitHandle : wait.work=some handle
  startBuffer : start.inputs=[dst] ∧ start.outputs=[dst]
  waitBuffer : wait.inputs=[dst] ∧ wait.outputs=[dst]
  stream : start.stream=wait.stream
  rank : start.rank=wait.rank
  device : start.device=wait.device
  begun : requests.started (abi.identity ⟨participant,[9,1,2]⟩) start
  pendingStart : requests.pending (abi.identity ⟨participant,[9,1,2]⟩) pending.native
  pendingWait : requests.pending (abi.identity ⟨participant,[9,1,2]⟩) reduced.native
  completed : requests.finished (abi.identity ⟨participant,[9,1,2]⟩) wait
  beginTick : Nat
  endTick : Nat
  beginBound : beginTick ≤ (objects.image start).length
  endBound : endTick ≤ (objects.image wait).length
  beginEvent : (objects.image start).requestStarted beginTick (abi.identity ⟨participant,[9,1,2]⟩)
  endEvent : (objects.image wait).requestFinished endTick (abi.identity ⟨participant,[9,1,2]⟩)

def CollectiveWork.token {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {requests}
    (work : CollectiveWork objects requests) : Nat := abi.identity ⟨work.participant,[9,1,2]⟩

/-- Asynchronous traffic may progress anywhere between the actual request
start and completion, including while an intervening LSE call runs. Only exact
same-participant source slots and their real image ticks establish the window. -/
structure WorkWindow {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    {requests : RequestRegistry σ} (work : CollectiveWork objects requests) (point : ExecutionPoint σ) : Prop where
  afterStart : LocalReady objects ledger ⟨work.start,work.beginTick⟩ point
  beforeCompletion : LocalReady objects ledger point ⟨work.wait,work.endTick⟩

/-- One-word provenance across actual local stores, consecutive host slots,
and the destination image's input read. The frame clauses mention addresses
only; the scalar transfer port remains the sole numerical move contract. -/
structure WordHandoff {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (registryOrder : RegisterOrder objects.registry objects.image)
    (value : Nat) (receiver : CallDescriptor σ) (tensor : Tensor) (index : Nat) where
  participant : Participant
  writer : Occurrence σ
  writerCalled : selection.invoked writer.call
  stored : σ
  store : trace.store writer value stored tensor index
  writerActive : objects.registry.active writer
  writerOpcode : objects.registry.opcode writer=some (.store value stored tensor index)
  ownerCall : (registryOrder.owner value).call=writer.call
  position : Fin (ledger.entries participant).length
  receiverPosition : Fin (ledger.entries participant).length
  receiverCalled : selection.invoked receiver
  writerLocation : (objects.exact writer.call writerCalled).location=((ledger.entries participant)[position]).1
  writerRecord : (objects.exact writer.call writerCalled).call=((ledger.entries participant)[position]).2
  receiverLocation : (objects.exact receiver receiverCalled).location=((ledger.entries participant)[receiverPosition]).1
  receiverRecord : (objects.exact receiver receiverCalled).call=((ledger.entries participant)[receiverPosition]).2
  follows : position.val < receiverPosition.val
  hostWrites : List ByteAddress
  hostSteps : LedgerSteps objects ledger events participant (position.val+1) receiverPosition.val hostWrites
  loadSite : Occurrence σ
  loadCall : loadSite.call=receiver
  loadedState : σ
  loadedValue : Nat
  load : trace.load loadSite loadedState tensor index loadedValue
  loadActive : objects.registry.active loadSite
  loadOpcode : objects.registry.opcode loadSite=some (.load loadedState tensor index)
  stableWriter : layout.buffer writer.call.after tensor.region=layout.buffer stored tensor.region
  stableReceiver : layout.buffer receiver.before tensor.region=layout.buffer stored tensor.region
  stableLoad : layout.buffer loadedState tensor.region=layout.buffer stored tensor.region
  writerOutside : ∀ tick,
    (objects.image writer.call).tick writer.coordinate writer.instruction+1≤tick →
    tick < (objects.image writer.call).length → ∀ byte : Fin 4,
    ((layout.buffer stored tensor.region).device,(layout.buffer stored tensor.region).base+4*tensor.index index+byte.val)
      ∉(objects.image writer.call).writes tick
  hostOutside : ∀ byte : Fin 4,
    ((layout.buffer stored tensor.region).device,(layout.buffer stored tensor.region).base+4*tensor.index index+byte.val)∉hostWrites
  receiverOutside : ∀ tick, tick < (objects.image receiver).tick loadSite.coordinate loadSite.instruction → ∀ byte : Fin 4,
    ((layout.buffer stored tensor.region).device,(layout.buffer stored tensor.region).base+4*tensor.index index+byte.val)
      ∉(objects.image receiver).writes tick

theorem WordHandoff.retained {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order value receiver tensor index observe}
    (handoff : WordHandoff objects ledger layout events trace order value receiver tensor index)
    (localMemory : ImageMemory loaded observe) (memory : ByteContracts events observe selection) (byte : Fin 4) :
    tensorByte layout observe handoff.loadedState tensor index byte=tensorByte layout observe handoff.stored tensor index byte := by
  let writer := objects.exact handoff.writer.call handoff.writerCalled
  let reader := objects.exact handoff.loadSite.call handoff.loadActive.1
  have stored := (writer.states.store handoff.writer.coordinate handoff.writer.instruction value handoff.stored tensor index
    handoff.writerActive.2 handoff.writerOpcode).1
  have writerBound := writer.states.bound handoff.writer.coordinate handoff.writer.instruction handoff.writerActive.2
  have wf := localMemory.suffix writer.object
    ((objects.image handoff.writer.call).tick handoff.writer.coordinate handoff.writer.instruction+1) (by omega)
    (layout.buffer handoff.stored tensor.region).device
    ((layout.buffer handoff.stored tensor.region).base+4*tensor.index index+byte.val)
    (fun tick lo hi => handoff.writerOutside tick lo hi byte)
  rw [writer.states.final,←stored] at wf
  have host := handoff.hostSteps.trace.frame memory _ _ (handoff.hostOutside byte)
  have writerEnd := writer.matching.after
  rw [handoff.writerRecord] at writerEnd
  have writerBoundary := writerEnd.trans (ledger.boundaries handoff.participant handoff.position).2
  have receiverStart := (objects.exact receiver handoff.receiverCalled).matching.before
  rw [handoff.receiverRecord] at receiverStart
  have receiverBoundary := receiverStart.trans (ledger.boundaries handoff.participant handoff.receiverPosition).1
  rw [←writerBoundary,←receiverBoundary] at host
  have loadedState := (reader.states.load handoff.loadSite.coordinate handoff.loadSite.instruction
    handoff.loadedState tensor index handoff.loadActive.2 handoff.loadOpcode).1
  have readBound := reader.states.bound handoff.loadSite.coordinate handoff.loadSite.instruction handoff.loadActive.2
  have rf := localMemory.interval reader.object 0
    ((objects.image handoff.loadSite.call).tick handoff.loadSite.coordinate handoff.loadSite.instruction)
    (Nat.zero_le _) (by omega)
    (layout.buffer handoff.stored tensor.region).device
    ((layout.buffer handoff.stored tensor.region).base+4*tensor.index index+byte.val)
    (by intro tick _ hi; rw [handoff.loadCall] at hi ⊢; exact handoff.receiverOutside tick hi byte)
  rw [reader.states.initial,←loadedState,handoff.loadCall] at rf
  unfold tensorByte
  rw [handoff.stableLoad]
  exact rf.trans (host.trans wf)

theorem WordHandoff.bytes {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order value receiver tensor index observe native readWord}
    (handoff : WordHandoff objects ledger layout events trace order value receiver tensor index)
    (api : PrimitiveContracts trace native readWord (tensorByte layout observe))
    (localMemory : ImageMemory loaded observe) (memory : ByteContracts events observe selection) (byte : Fin 4) :
    tensorByte layout observe handoff.loadedState tensor index byte=
      byteOf (if tensor.isNeg then flipSign (trace.bits value) else trace.bits value) byte :=
  (handoff.retained localMemory memory byte).trans (api.store _ _ _ _ _ handoff.store byte)

/-- Register production, the actual store, exact host slots and the actual
receiver load form a causal path, independently of any abstract epoch. -/
theorem WordHandoff.owner_before_load {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order value receiver tensor index}
    (handoff : WordHandoff objects ledger layout events trace order value receiver tensor index)
    (requests : RequestRegistry σ) :
    ExecutionPath objects ledger requests (instructionPoint objects (order.owner value))
      (instructionPoint objects handoff.loadSite) := by
  have earlier := order.earlier handoff.writer (.store value handoff.stored tensor index)
    handoff.writerActive handoff.writerOpcode value (by simp [Opcode.operands]) handoff.ownerCall
  have owner := same_call_before objects ledger requests (order.owner value) handoff.writer
    handoff.writerActive handoff.ownerCall earlier
  have writerBound := (objects.exact handoff.writer.call handoff.writerCalled).states.bound
    handoff.writer.coordinate handoff.writer.instruction handoff.writerActive.2
  have loadBound := (objects.exact handoff.loadSite.call handoff.loadActive.1).states.bound
    handoff.loadSite.coordinate handoff.loadSite.instruction handoff.loadActive.2
  let writerPoint : LedgerPoint objects ledger handoff.participant (completedPoint objects handoff.writer) :=
    { position := handoff.position, called := handoff.writerCalled,
      location := handoff.writerLocation, record := handoff.writerRecord, bound := by
        change (objects.image handoff.writer.call).tick handoff.writer.coordinate handoff.writer.instruction+1 ≤ (objects.image handoff.writer.call).length
        omega }
  let loadPoint : LedgerPoint objects ledger handoff.participant
      ⟨receiver,(objects.image receiver).tick handoff.loadSite.coordinate handoff.loadSite.instruction⟩ :=
    { position := handoff.receiverPosition, called := handoff.receiverCalled,
      location := handoff.receiverLocation, record := handoff.receiverRecord,
      bound := by simpa only [handoff.loadCall] using Nat.le_of_lt loadBound }
  have bridge : LocalReady objects ledger (completedPoint objects handoff.writer) (instructionPoint objects handoff.loadSite) := by
    unfold instructionPoint
    rw [handoff.loadCall]
    exact .host handoff.participant _ _ writerPoint loadPoint handoff.follows
  exact (ExecutionPath.trans owner
    (instruction_advance objects ledger requests handoff.writer handoff.writerActive)).then_ready bridge

inductive SendKind where
  | loss | lse
  deriving DecidableEq

/-- An injection is the actual peer-local loss/LSE tensor, read by that peer's
exact start/gather call at one row. It is not an unconstrained peer value. -/
structure PeerSend {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (value : Nat) where
  rank : Nat
  row : Nat
  kind : SendKind
  descriptor : CallDescriptor σ
  called : selection.invoked descriptor
  location : (objects.exact descriptor called).location=
    ⟨abi.peerParticipant rank,if kind=.loss then [9,1,2] else [9,1,1]⟩
  callKind : (objects.exact descriptor called).call.kind=if kind=.loss then .reduceStart else .gather
  tensor : Tensor
  exactInput : descriptor.inputs=[tensor]
  rowBound : row < tensor.shape.prod
  rankEq : descriptor.rank=rank
  handoff : WordHandoff objects ledger layout events trace order value descriptor tensor row

/-- Rank transfers admit both exact peer injections and intermediate partial
sums. Internal partial sums must be earlier dynamic operands in the SAME
communicator/sequence; no algorithm (ring/tree/CollNet/NVLS/network) is capped. -/
inductive RankOrigin {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (value row : Nat) (sourceInstruction : Occurrence σ) (sourceWork : CollectiveWork objects requests) : Prop
  | injection (send : PeerSend objects ledger layout events trace order value)
      (loss : send.kind=.loss) (coordinate : send.row=row)
      (rank : send.rank=sourceWork.start.rank)
      (descriptor : send.descriptor=sourceWork.start) (read : sourceInstruction=send.handoff.loadSite) : RankOrigin objects ledger layout events trace order requests value row sourceInstruction sourceWork
  | partialSum (read : sourceInstruction=order.owner value) (owner : (order.owner value).call=sourceWork.wait)
      (coordinate : (objects.image (order.owner value).call).logicalIndex (order.owner value).coordinate (order.owner value).instruction=row) :
      RankOrigin objects ledger layout events trace order requests value row sourceInstruction sourceWork

structure RankRoute {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (site : Occurrence σ) (value channel sequence sourceRank destinationRank : Nat) where
  sender : CollectiveWork objects requests
  receiver : CollectiveWork objects requests
  sourceRankEq : sender.wait.rank=sourceRank
  destinationRankEq : receiver.wait.rank=destinationRank
  destinationCall : site.call=receiver.wait
  communicator : requests.communicator sender.token=requests.communicator receiver.token
  sourceSequence : requests.sequence sender.token=sequence
  destinationSequence : requests.sequence receiver.token=sequence
  group : sender.group=receiver.group
  row : Nat
  coordinate : (objects.image site.call).logicalIndex site.coordinate site.instruction=row
  sourceInstruction : Occurrence σ
  origin : RankOrigin objects ledger layout events trace order requests value row sourceInstruction sender
  packet : TransportPacket
  packetSender : packet.senderToken=sender.token
  packetReceiver : packet.receiverToken=receiver.token
  packetChannel : packet.channel=channel
  packetSequence : packet.sequence=sequence
  packetSource : packet.sourceRank=sourceRank
  packetDestination : packet.destinationRank=destinationRank
  packetRow : packet.row=row
  wire : WireDelivery objects ledger requests sourceInstruction site packet
  sendWindow : WorkWindow objects ledger sender wire.send
  receiveWindow : WorkWindow objects ledger receiver wire.receive

/-- All-gather injections carry LSE words from the exact peer gather input.
They precede the associated reduction start and use their own NCCL sequence. -/
structure GatherRoute {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (site : Occurrence σ) (value channel sequence sourceRank destinationRank : Nat) where
  sender : CollectiveWork objects requests
  receiver : CollectiveWork objects requests
  send : PeerSend objects ledger layout events trace order value
  lse : send.kind=.lse
  sourceRankEq : send.rank=sourceRank
  sourceParticipant : sender.participant=abi.peerParticipant sourceRank
  destinationRankEq : site.call.rank=destinationRank
  destinationCalled : selection.invoked site.call
  destinationLocation : (objects.exact site.call destinationCalled).location=⟨receiver.participant,[9,1,1]⟩
  destinationKind : (objects.exact site.call destinationCalled).call.kind=.gather
  destinationOutput : site.call.outputs=[receiver.gathered]
  world : Nat
  rows : Nat
  shape : receiver.gathered.shape=[world,rows]
  sourceBound : sourceRank < world
  rowBound : send.row < rows
  coordinate : (objects.image site.call).logicalIndex site.coordinate site.instruction=send.row
  storeSite : Occurrence σ
  storeCall : storeSite.call=site.call
  stored : σ
  storedActive : objects.registry.active storeSite
  storedOpcode : objects.registry.opcode storeSite=some
    (.store (objects.registry.output site) stored receiver.gathered (sourceRank*rows+send.row))
  group : send.descriptor.group=receiver.group ∧ site.call.group=receiver.group ∧ sender.group=receiver.group
  communicator : requests.communicator sender.token=requests.communicator receiver.token
  sourceSequence : requests.gatherSequence sender.token=sequence
  destinationSequence : requests.gatherSequence receiver.token=sequence
  packet : TransportPacket
  packetSender : packet.senderToken=sender.token
  packetReceiver : packet.receiverToken=receiver.token
  packetChannel : packet.channel=channel
  packetSequence : packet.sequence=sequence
  packetSource : packet.sourceRank=sourceRank
  packetDestination : packet.destinationRank=destinationRank
  packetRow : packet.row=send.row
  wire : WireDelivery objects ledger requests send.handoff.loadSite site packet
  sendCall : wire.send.call=send.descriptor
  receiveCall : wire.receive.call=site.call

theorem RankOrigin.owner_before_source {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests value row sourceInstruction sourceWork}
    (origin : RankOrigin objects ledger layout events trace order requests value row sourceInstruction sourceWork) :
    instructionPoint objects (order.owner value)=instructionPoint objects sourceInstruction ∨
      ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects sourceInstruction) := by
  cases origin with
  | injection send loss coordinate rank descriptor read =>
    right
    simpa only [read] using send.handoff.owner_before_load requests
  | partialSum read owner coordinate =>
    left
    exact congrArg (instructionPoint objects) read.symm

theorem RankRoute.before {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests site value channel sequence sourceRank destinationRank}
    (route : RankRoute objects ledger layout events trace order requests site value channel sequence sourceRank destinationRank) :
    ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) := by
  rcases route.origin.owner_before_source with same | previous
  · simpa only [same] using route.wire.before
  · exact .trans previous route.wire.before

theorem GatherRoute.before {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests site value channel sequence sourceRank destinationRank}
    (route : GatherRoute objects ledger layout events trace order requests site value channel sequence sourceRank destinationRank) :
    ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) :=
  .trans (route.send.handoff.owner_before_load requests) route.wire.before

/-- Cross-call local memory transport consumes the actual receiver load
strictly before its transfer instruction, in addition to byte/CTA identities. -/
structure GlobalMove {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (site : Occurrence σ)
    (value writeEpoch readEpoch address sourceCTA destinationCTA : Nat) where
  tensor : Tensor
  index : Nat
  handoff : WordHandoff objects ledger layout events trace order value site.call tensor index
  addressEq : address=(layout.buffer handoff.loadedState tensor.region).base+4*tensor.index index
  writeEpochEq : writeEpoch=(objects.image handoff.writer.call).memoryEpoch handoff.writer.coordinate handoff.writer.instruction
  readEpochEq : readEpoch=(objects.image handoff.loadSite.call).memoryEpoch handoff.loadSite.coordinate handoff.loadSite.instruction
  sourceCTAeq : sourceCTA=(objects.image handoff.writer.call).cta handoff.writer.coordinate handoff.writer.instruction
  destinationCTAeq : destinationCTA=(objects.image handoff.loadSite.call).cta handoff.loadSite.coordinate handoff.loadSite.instruction
  loadBefore : (objects.image handoff.loadSite.call).tick handoff.loadSite.coordinate handoff.loadSite.instruction <
    (objects.image site.call).tick site.coordinate site.instruction

def LocalMoveAllowed {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (site : Occurrence σ) (value : Nat) : Transfer → Prop
  | .shuffle _ _ _ | .shared _ _ _ _ _ => (order.owner value).call=site.call
  | .global writeEpoch readEpoch address sourceCTA destinationCTA =>
    (order.owner value).call=site.call ∨
      Nonempty (GlobalMove objects ledger layout events trace order site value writeEpoch readEpoch address sourceCTA destinationCTA)
  | .rank _ _ _ _ => False

theorem LocalMoveAllowed.before {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order site value edge}
    (allowed : LocalMoveAllowed objects ledger layout events trace order site value edge) (requests : RequestRegistry σ)
    (active : objects.registry.active site) (decoded : objects.registry.opcode site=some (.transfer edge value)) :
    ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) := by
  have localPath : (order.owner value).call=site.call →
      ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) := by
    intro same
    exact same_call_before objects ledger requests (order.owner value) site active same
      (order.earlier site (.transfer edge value) active decoded value (by simp [Opcode.operands]) same)
  cases edge with
  | shuffle a b c => exact localPath allowed
  | shared a b c d e => exact localPath allowed
  | global a b c d e =>
    rcases allowed with same | witness
    · exact localPath same
    · exact witness.elim (fun move => .trans (move.handoff.owner_before_load requests)
        (same_call_before objects ledger requests move.handoff.loadSite site active move.handoff.loadCall move.loadBefore))
  | rank a b c d => cases allowed

/-- Only the generated local/host/message graph supplies operand causality.
Its explicit runtime acyclicity law imposes no global serialization. -/
structure TransferProtocol {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ) where
  history : CausalHistory objects ledger requests
  waits : ∀ d (called : selection.invoked d), (objects.exact d called).call.kind=.reduceWait →
    ∃ work : CollectiveWork objects requests, work.wait=d
  ranks : ∀ s channel sequence sourceRank destinationRank value,
    objects.registry.active s → objects.registry.opcode s=some (.transfer (.rank channel sequence sourceRank destinationRank) value) →
    Nonempty (RankRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank) ∨
      Nonempty (GatherRoute objects ledger layout events trace order requests s value channel sequence sourceRank destinationRank)
  localMoves : ∀ s edge value, objects.registry.active s → objects.registry.opcode s=some (.transfer edge value) →
    (∀ channel sequence sourceRank destinationRank, edge≠Transfer.rank channel sequence sourceRank destinationRank) →
    LocalMoveAllowed objects ledger layout events trace order s value edge

theorem TransferProtocol.operand_before {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests}
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (site : Occurrence σ) (op : Opcode σ) (active : objects.registry.active site)
    (decoded : objects.registry.opcode site=some op) (value : Nat) (member : value∈op.operands) :
    ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) := by
  classical
  by_cases transfer : ∃ edge operand, op=Opcode.transfer edge operand
  · rcases transfer with ⟨edge,operand,rfl⟩
    have same : value=operand := by simpa only [Opcode.operands,List.mem_singleton] using member
    subst value
    cases edge with
    | rank channel sequence sourceRank destinationRank =>
      rcases protocol.ranks site channel sequence sourceRank destinationRank operand active decoded with route | route
      · exact route.elim (fun route => route.before)
      · exact route.elim (fun route => route.before)
    | shuffle a b c =>
      exact (protocol.localMoves site (.shuffle a b c) operand active decoded (by intro _ _ _ _ h; cases h)).before requests active decoded
    | shared a b c d e =>
      exact (protocol.localMoves site (.shared a b c d e) operand active decoded (by intro _ _ _ _ h; cases h)).before requests active decoded
    | global a b c d e =>
      exact (protocol.localMoves site (.global a b c d e) operand active decoded (by intro _ _ _ _ h; cases h)).before requests active decoded
  · have same := order.localOperands site op active decoded (by
      intro edge operand equal
      exact transfer ⟨edge,operand,equal⟩) value member
    exact same_call_before objects ledger requests (order.owner value) site active same
      (order.earlier site op active decoded value member same)

theorem TransferProtocol.operands {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests}
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (site : Occurrence σ) (op : Opcode σ) (active : objects.registry.active site)
    (decoded : objects.registry.opcode site=some op) : ∀ value∈op.operands,
    objects.registry.active (order.owner value) ∧ objects.registry.output (order.owner value)=value ∧
      ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site) := by
  intro value member
  exact ⟨(order.operands site op active decoded value member).1,
    (order.operands site op active decoded value member).2,protocol.operand_before site op active decoded value member⟩

/-- The loader consumes real operand paths and the absence of a causal cycle
at this instruction, along with exact request and transfer evidence. -/
def StepPermit {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (layout : Layout σ) (events : ByteEvents σ) (trace : PrimitiveTrace σ)
    (order : RegisterOrder objects.registry objects.image) (requests : RequestRegistry σ)
    (site : Occurrence σ) (op : Opcode σ) : Prop :=
  (∀ called, (objects.exact site.call called).call.kind=.reduceWait →
    ∃ work : CollectiveWork objects requests, work.wait=site.call) ∧
  (¬ExecutionPath objects ledger requests (instructionPoint objects site) (instructionPoint objects site)) ∧
  (∀ value∈op.operands, objects.registry.active (order.owner value) ∧
    objects.registry.output (order.owner value)=value ∧
      ExecutionPath objects ledger requests (instructionPoint objects (order.owner value)) (instructionPoint objects site)) ∧
  (op.produces → order.owner (objects.registry.output site)=site) ∧
  match op with
  | .transfer (.rank channel sequence sourceRank destinationRank) value =>
    Nonempty (RankRoute objects ledger layout events trace order requests site value channel sequence sourceRank destinationRank) ∨
      Nonempty (GatherRoute objects ledger layout events trace order requests site value channel sequence sourceRank destinationRank)
  | .transfer edge value => LocalMoveAllowed objects ledger layout events trace order site value edge
  | _ => ∀ value∈op.operands, (order.owner value).call=site.call

theorem TransferProtocol.permits {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger layout events trace order requests}
    (protocol : TransferProtocol objects ledger layout events trace order requests)
    (site : Occurrence σ) (op : Opcode σ) (active : objects.registry.active site)
    (decoded : objects.registry.opcode site=some op) : StepPermit objects ledger layout events trace order requests site op := by
  refine ⟨protocol.waits site.call,protocol.history.noCycle _,protocol.operands site op active decoded,
    order.producer site op active decoded,?_⟩
  cases op with
  | transfer edge value =>
    cases edge with
    | rank channel sequence sourceRank destinationRank =>
      exact protocol.ranks site channel sequence sourceRank destinationRank value active decoded
    | shuffle a b c => exact protocol.localMoves site (.shuffle a b c) value active decoded (by intro _ _ _ _ h; cases h)
    | shared a b c d e => exact protocol.localMoves site (.shared a b c d e) value active decoded (by intro _ _ _ _ h; cases h)
    | global a b c d e => exact protocol.localMoves site (.global a b c d e) value active decoded (by intro _ _ _ _ h; cases h)
  | raw x => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | load state tensor index => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | literal x => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | natural x => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | negativeInfinity => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | binary op a b => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | unary op a => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | fma a b c => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | hostScale scale a => exact order.localOperands site _ active decoded (by intro edge value h; cases h)
  | store value state tensor index => exact order.localOperands site _ active decoded (by intro edge value h; cases h)

theorem different_start_different_token {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {requests}
    (a b : CollectiveWork objects requests) (different : a.participant≠b.participant) : a.token≠b.token := by
  intro same
  have location : (⟨a.participant,[9,1,2]⟩ : SourceLocation)=⟨b.participant,[9,1,2]⟩ := abi.injective same
  exact different (congrArg SourceLocation.participant location)

#axiomsClean TransferProtocol.operand_before
#axiomsClean WordHandoff.owner_before_load
#axiomsClean RankRoute.before
#axiomsClean GatherRoute.before
#axiomsClean TransferProtocol.permits
#axiomsClean WordHandoff.retained
#axiomsClean WordHandoff.bytes
#axiomsClean different_start_different_token
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
