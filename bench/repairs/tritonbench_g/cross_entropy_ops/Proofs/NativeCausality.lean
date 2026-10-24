import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeLedger
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CausalExecution

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton
open CrossEntropy1.Floating CrossEntropy1.Physical
noncomputable section

def instructionPoint {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (s : Occurrence σ) : ExecutionPoint σ :=
  ⟨s.call,(objects.image s.call).tick s.coordinate s.instruction⟩

def completedPoint {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (s : Occurrence σ) : ExecutionPoint σ :=
  ⟨s.call,(objects.image s.call).tick s.coordinate s.instruction+1⟩

/-- The local point belongs to one exact slot in the complete host ledger. -/
structure LedgerPoint {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (participant : Participant) (point : ExecutionPoint σ) where
  position : Fin (ledger.entries participant).length
  called : selection.invoked point.call
  location : (objects.exact point.call called).location=((ledger.entries participant)[position]).1
  record : (objects.exact point.call called).call=((ledger.entries participant)[position]).2
  bound : point.tick ≤ (objects.image point.call).length

/-- Non-strict local readiness is either actual image-boundary order or
order of exact source slots on the same participant. Parallel participants
are never compared by this relation. -/
inductive LocalReady {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source) :
    ExecutionPoint σ → ExecutionPoint σ → Prop
  | same (d : CallDescriptor σ) (called : selection.invoked d) (first last : Nat)
      (ordered : first ≤ last) (bound : last ≤ (objects.image d).length) :
      LocalReady objects ledger ⟨d,first⟩ ⟨d,last⟩
  | host (participant : Participant) (a b : ExecutionPoint σ)
      (first : LedgerPoint objects ledger participant a) (last : LedgerPoint objects ledger participant b)
      (ordered : first.position.val < last.position.val) : LocalReady objects ledger a b

/-- The send and receive are logged at these actual loaded-image boundaries,
with the same packet and dynamic source-register identity. Delivered is the
ordinary runtime's matched-message fact, not an arbitrary ordering function. -/
structure MessageEdge {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (requests : RequestRegistry σ)
    (packet : TransportPacket) (send receive : ExecutionPoint σ) : Prop where
  sendCalled : selection.invoked send.call
  receiveCalled : selection.invoked receive.call
  sendBound : send.tick ≤ (objects.image send.call).length
  receiveBound : receive.tick ≤ (objects.image receive.call).length
  sent : (objects.image send.call).sent send.tick packet
  received : (objects.image receive.call).received receive.tick packet
  delivered : requests.delivered packet
  sourceRank : send.call.rank=packet.sourceRank
  destinationRank : receive.call.rank=packet.destinationRank
  sourceCommunicator : requests.communicator packet.senderToken=packet.communicator
  destinationCommunicator : requests.communicator packet.receiverToken=packet.communicator

/-- Every causal edge is an actual local/host advance or a matched delivery.
There is no independently supplied happens-before relation. -/
inductive ExecutionEdge {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) : ExecutionPoint σ → ExecutionPoint σ → Prop
  | local {a b} (ready : LocalReady objects ledger a b) (different : a≠b) : ExecutionEdge objects ledger requests a b
  | message {a b} (packet : TransportPacket) (event : MessageEdge objects requests packet a b) :
      ExecutionEdge objects ledger requests a b

inductive ExecutionPath {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) : ExecutionPoint σ → ExecutionPoint σ → Prop
  | edge {a b} : ExecutionEdge objects ledger requests a b → ExecutionPath objects ledger requests a b
  | trans {a b c} : ExecutionPath objects ledger requests a b → ExecutionPath objects ledger requests b c →
      ExecutionPath objects ledger requests a c

/-- Acyclicity of the actual event graph is the ordinary runtime causality
law. It asserts no result and no total order of independent parallel work. -/
structure CausalHistory {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) : Prop where
  noCycle : ∀ point, ¬ExecutionPath objects ledger requests point point

theorem LocalReady.path_or_equal {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests a b}
    (ready : LocalReady objects ledger a b) : a=b ∨ ExecutionPath objects ledger requests a b := by
  classical
  by_cases same : a=b
  · exact Or.inl same
  · exact Or.inr (.edge (.local ready same))

theorem ExecutionPath.then_ready {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests a b c}
    (path : ExecutionPath objects ledger requests a b) (ready : LocalReady objects ledger b c) :
    ExecutionPath objects ledger requests a c := by
  rcases ready.path_or_equal (requests:=requests) with same | next
  · simpa only [same] using path
  · exact .trans path next

theorem ExecutionPath.ready_then {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests a b c}
    (ready : LocalReady objects ledger a b) (path : ExecutionPath objects ledger requests b c) :
    ExecutionPath objects ledger requests a c := by
  rcases ready.path_or_equal (requests:=requests) with same | previous
  · simpa only [same] using path
  · exact .trans previous path

theorem instruction_advance {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) (s : Occurrence σ) (active : objects.registry.active s) :
    ExecutionPath objects ledger requests (instructionPoint objects s) (completedPoint objects s) := by
  have bound := (objects.exact s.call active.1).states.bound s.coordinate s.instruction active.2
  refine .edge (.local (.same s.call active.1 _ _ (Nat.le_succ _) (by omega)) ?_)
  intro same
  have impossible := congrArg ExecutionPoint.tick same
  simp only [instructionPoint,completedPoint] at impossible
  omega

theorem same_call_before {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) (producer consumer : Occurrence σ)
    (active : objects.registry.active consumer) (same : producer.call=consumer.call)
    (earlier : (objects.image producer.call).tick producer.coordinate producer.instruction <
      (objects.image consumer.call).tick consumer.coordinate consumer.instruction) :
    ExecutionPath objects ledger requests (instructionPoint objects producer) (instructionPoint objects consumer) := by
  have bound := (objects.exact consumer.call active.1).states.bound consumer.coordinate consumer.instruction active.2
  have ready : LocalReady objects ledger (instructionPoint objects producer) (instructionPoint objects consumer) := by
    unfold instructionPoint
    rw [same] at earlier ⊢
    exact .same consumer.call active.1 _ _ (Nat.le_of_lt earlier) (Nat.le_of_lt bound)
  refine .edge (.local ready ?_)
  intro equal
  have ticks := congrArg ExecutionPoint.tick equal
  simp only [instructionPoint] at ticks
  omega

/-- A concrete packet must be sent after its actual producer has completed,
and arrive before the consuming instruction. Endpoint readiness follows the
selected image clocks or exact source-ledger slots. -/
structure WireDelivery {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded) (ledger : HostLedger source)
    (requests : RequestRegistry σ) (producer consumer : Occurrence σ) (packet : TransportPacket) where
  send : ExecutionPoint σ
  receive : ExecutionPoint σ
  sourceActive : objects.registry.active producer
  destinationActive : objects.registry.active consumer
  payload : packet.payload=objects.registry.output producer
  message : MessageEdge objects requests packet send receive
  sourceReady : LocalReady objects ledger (completedPoint objects producer) send
  destinationReady : LocalReady objects ledger receive (instructionPoint objects consumer)

theorem WireDelivery.before {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests producer consumer packet}
    (wire : WireDelivery objects ledger requests producer consumer packet) :
    ExecutionPath objects ledger requests (instructionPoint objects producer) (instructionPoint objects consumer) := by
  have started := (instruction_advance objects ledger requests producer wire.sourceActive).then_ready wire.sourceReady
  exact (ExecutionPath.trans started (.edge (.message packet wire.message))).then_ready wire.destinationReady

theorem CausalHistory.reject_reverse_local {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests}
    (history : CausalHistory objects ledger requests) (producer consumer : Occurrence σ)
    (producerActive : objects.registry.active producer) (same : producer.call=consumer.call)
    (future : (objects.image consumer.call).tick consumer.coordinate consumer.instruction <
      (objects.image producer.call).tick producer.coordinate producer.instruction)
    (path : ExecutionPath objects ledger requests (instructionPoint objects producer) (instructionPoint objects consumer)) : False := by
  have back := same_call_before objects ledger requests consumer producer producerActive same.symm future
  exact history.noCycle _ (.trans path back)

#axiomsClean WireDelivery.before
#axiomsClean CausalHistory.reject_reverse_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
