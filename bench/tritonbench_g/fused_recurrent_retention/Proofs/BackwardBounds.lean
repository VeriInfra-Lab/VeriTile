import bench.tritonbench_g.fused_recurrent_retention.Proofs.SafetyAccess
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardReal

/-! Allocation bounds for the exact active backward accesses. Tail lanes
require no storage; the seed allocation is required only when enabled. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
open VeriTile.Triton

structure Bounds (p : Layout) (l : Launch) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) : Prop where
  query : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyAddress p l t k.val < bounds r.q
  key : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyAddress p l t k.val < bounds r.k
  value : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p l j.val →
    valueAddress p l t j.val < bounds r.v
  dout : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p l j.val →
    valueAddress p l t j.val < bounds r.do_
  dq : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyGradientAddress p l t k.val < bounds r.dq
  dk : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p l k.val →
    keyGradientAddress p l t k.val < bounds r.dk
  dv : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p l j.val →
    outputAddress p l t j.val < bounds r.dv
  seed : hasSeed = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p l k.val → activeValue p l j.val → stateAddress p l k.val j.val < bounds r.initial

end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
