import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Model

/-! Small consequence of the ordinary block-barrier/release/acquire API.
These constructors state the documented primitive ordering rules, not GPU
implementation theorems or numerical assumptions. All producer lanes occur
before the scalar release through the CTA barrier; all consumer lanes follow
the acquired CAS through its compiler-generated broadcast barrier. The checked
PTX evidence binds both barriers to the actual ordinary wrapper specialization.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.Publication

inductive Point where
  | producerStore (lane : Nat)
  | producerBarrierArrival (lane : Nat)
  | producerBarrierDeparture
  | release
  | acquire
  | consumerBarrierDeparture (lane : Nat)
  | consumerLoad (lane : Nat)

inductive Before : Point → Point → Prop where
  | storeArrival (lane) : Before (.producerStore lane) (.producerBarrierArrival lane)
  | blockSync (lane) : Before (.producerBarrierArrival lane) .producerBarrierDeparture
  | barrierRelease : Before .producerBarrierDeparture .release
  | releaseAcquire : Before .release .acquire
  | acquireBroadcast (lane) : Before .acquire (.consumerBarrierDeparture lane)
  | broadcastLoad (lane) : Before (.consumerBarrierDeparture lane) (.consumerLoad lane)
  | trans : Before a b → Before b c → Before a c

/-- The synchronization covers every producing lane, including warp 1/tid 32
from the original counterexample. It is not limited to the publishing thread. -/
theorem every_lane_published_before_read (producerLane consumerLane : Nat) :
    Before (.producerStore producerLane) (.consumerLoad consumerLane) :=
  .trans (.storeArrival producerLane)
    (.trans (.blockSync producerLane)
      (.trans .barrierRelease
        (.trans .releaseAcquire
          (.trans (.acquireBroadcast consumerLane) (.broadcastLoad consumerLane)))))

#axiomsClean every_lane_published_before_read
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Publication
