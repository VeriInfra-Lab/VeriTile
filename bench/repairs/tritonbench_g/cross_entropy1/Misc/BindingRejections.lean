import bench.repairs.tritonbench_g.cross_entropy1.Misc.HostFPEquiv

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating.Revision005
open Source Host Physical
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- CE1-R3's fixed-object multiply-to-add retag is rejected at the same
occurrence, even if all descriptor strings/configuration remain unchanged. -/
theorem reject_multiply_to_add {source abi selection loaded trace}
    (objects : ObjectSelection (σ:=Nat) source abi selection loaded)
    (events : TraceImage objects.registry trace) (site : Occurrence Nat) (out : Nat)
    (actual : (objects.image site.call).opcode site.coordinate site.instruction=some (.binary .mul 11 12)) :
    ¬trace.binary site .add 11 12 out := by
  intro retag
  have h := events.event site (.binary .add 11 12) out retag
  have clash := actual.symm.trans h.2.1
  cases clash

/-- CE1-R4's store at 7 cannot be labelled as a store whose exact dynamic
post-state is 1. A stable buffer and nil trace at 7 cannot repair the failure. -/
theorem reject_store_state_seven {source abi selection loaded trace}
    (objects : ObjectSelection (σ:=Nat) source abi selection loaded)
    (events : TraceImage objects.registry trace) (site : Occurrence Nat) (value index : Nat) (tensor : Tensor)
    (actual : (objects.image site.call).state ((objects.image site.call).tick site.coordinate site.instruction+1)=1) :
    ¬trace.store site value 7 tensor index := by
  intro relabel
  have h := events.event site (.store value 7 tensor index) 0 relabel
  have state := ((objects.exact site.call h.1.1).states.store site.coordinate site.instruction value 7 tensor index h.1.2 h.2.1).1
  have clash : (7:Nat)=1 := state.trans actual
  omega

/-- A future producer is rejected using actual image ticks, without any
register epoch assignment. This applies to every ordinary binary operand. -/
theorem reject_future_operand {σ : Type} {registry : InstructionRegistry σ}
    {image : CallDescriptor σ → CallImage σ}
    (order : RegisterOrder registry image) (site : Occurrence σ) (op : ScalarBinary) (a b : Nat)
    (active : registry.active site) (decoded : registry.opcode site=some (.binary op a b))
    (same : (order.owner a).call=site.call)
    (future : (image site.call).tick site.coordinate site.instruction ≤
      (image (order.owner a).call).tick (order.owner a).coordinate (order.owner a).instruction) : False := by
  have h := order.earlier site (.binary op a b) active decoded a (by simp [Opcode.operands]) same
  exact Nat.not_lt_of_ge future h

/-- The reviewer's exact load-at-8/unary-at-2 counterexample fails even if
all the original active/owner/load-state metadata and opcode identities hold. -/
theorem reject_load_eight_unary_two {source abi selection loaded}
    (objects : ObjectSelection (σ:=Nat) source abi selection loaded)
    (order : RegisterOrder objects.registry objects.image) (producer consumer : Occurrence Nat)
    (tensor : Tensor)
    (_load : objects.registry.opcode producer=some (.load 8 tensor 0))
    (active : objects.registry.active consumer)
    (decoded : objects.registry.opcode consumer=some (.unary .neg 11))
    (owner : order.owner 11=producer) (same : producer.call=consumer.call)
    (producerTick : (objects.image producer.call).tick producer.coordinate producer.instruction=8)
    (consumerTick : (objects.image consumer.call).tick consumer.coordinate consumer.instruction=2) : False := by
  have h := order.earlier consumer (.unary .neg 11) active decoded 11 (by simp [Opcode.operands]) (by simpa only [owner] using same)
  rw [owner,producerTick,consumerTick] at h
  omega

/-- A cross-call transfer at tick 2 cannot claim its receiver load at tick 8. -/
theorem reject_receiver_load_eight_transfer_two {source abi selection loaded}
    {objects : ObjectSelection (σ:=Nat) source abi selection loaded} {ledger layout events trace order site value writeEpoch readEpoch address sourceCTA destinationCTA}
    (move : GlobalMove objects ledger layout events trace order site value writeEpoch readEpoch address sourceCTA destinationCTA)
    (loadTick : (objects.image move.handoff.loadSite.call).tick move.handoff.loadSite.coordinate move.handoff.loadSite.instruction=8)
    (consumerTick : (objects.image site.call).tick site.coordinate site.instruction=2) : False := by
  have h := move.loadBefore
  rw [loadTick,consumerTick] at h
  omega

/-- Opposite packet dependency paths cannot form a cycle. This compares
actual message edges and local events, never clocks of different ranks. -/
theorem reject_cyclic_rank_exchange {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests a b packetAB packetBA}
    (history : CausalHistory objects ledger requests)
    (ab : WireDelivery objects ledger requests a b packetAB)
    (ba : WireDelivery objects ledger requests b a packetBA) : False :=
  history.noCycle _ (.trans ab.before ba.before)

/-- A valid packet witness cannot make a later same-image source available
to an earlier consumer, including via a spurious cross-rank detour. -/
theorem reject_packet_future_source {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded} {ledger requests producer consumer packet}
    (history : CausalHistory objects ledger requests)
    (wire : WireDelivery objects ledger requests producer consumer packet)
    (same : producer.call=consumer.call)
    (future : (objects.image consumer.call).tick consumer.coordinate consumer.instruction <
      (objects.image producer.call).tick producer.coordinate producer.instruction) : False :=
  history.reject_reverse_local producer consumer wire.sourceActive same future wire.before

/-- Same runtime and same descriptor metadata cannot replace the forward
launch slot with a later host-add slot under the same invocation identity. -/
theorem reject_other_source_slot {σ : Type} (abi : InvocationABI σ) :
    abi.identity ⟨.main,[8]⟩≠abi.identity ⟨.main,[9,2]⟩ := by
  intro same
  have clash := abi.injective same
  cases clash

/-- A wait on the same numeric handle still cannot use a stale/future native
state instead of this exact start→LSE→wait chain's reduced state. -/
theorem reject_wait_state_seven {source abi selection loaded requests}
    {objects : ObjectSelection (σ:=Nat) source abi selection loaded}
    (work : CollectiveWork objects requests) (actual : work.reduced.native=1) : work.wait.before≠7 := by
  intro stale
  have clash : (7:Nat)=1 := stale.symm.trans (work.waitBefore.trans actual)
  omega

#axiomsClean reject_multiply_to_add
#axiomsClean reject_store_state_seven
#axiomsClean reject_future_operand
#axiomsClean reject_load_eight_unary_two
#axiomsClean reject_receiver_load_eight_transfer_two
#axiomsClean reject_cyclic_rank_exchange
#axiomsClean reject_packet_future_source
#axiomsClean reject_other_source_slot
#axiomsClean reject_wait_state_seven
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating.Revision005
