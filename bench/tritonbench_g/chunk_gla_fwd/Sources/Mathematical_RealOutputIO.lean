import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_IO
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealOutputFlat

/-! Explicit output-consumer IO. The formula reads Q/V/G/H/A from initial
memory, independently of the implementation and its accumulators. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO
open VeriTile.Triton RealIO
open scoped BigOperators RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

structure Layout where
  keyHead : Nat
  keyTime : Nat
  valueHead : Nat
  valueTime : Nat
  historyHead : Nat
  historyKey : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
structure Regions where
  query : RegionName
  value : RegionName
  gate : RegionName
  history : RegionName
  output : RegionName
  attention : RegionName
structure Launch where
  value : Nat
  chunk : Nat
  head : Nat
structure Inputs where
  query : Nat → ℝ
  value : Nat → ℝ
  gate : Nat → ℝ
  history : Nat → ℝ
  attention : Nat → ℝ

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) :=
  l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)
def active (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) : Prop :=
  l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V

/-- Full K-block gated history contraction plus causal A·V, with zero extension
outside the source block pointers' checked parents. -/
def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  let t := l.chunk*p.BT+i.1.val
  let v := l.value*p.BV+i.2.1.val
  (∑ kb : Fin ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb.val*p.BK+e.val
    (if t < p.T ∧ k < p.K then
      a.query (l.head*p.keyHead+t*p.keyTime+k) * scale * Real.exp (a.gate (l.head*p.keyHead+t*p.keyTime+k)) else 0) *
    (if k < p.K ∧ v < p.V then
      a.history (l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+v) else 0)) +
  ∑ j : Fin p.BT,
    (if j.val ≤ i.1.val then (if t < p.T then a.attention (l.head*p.T*p.BT+t*p.BT+j.val) else 0) else 0) *
    (if l.chunk*p.BT+j.val < p.T ∧ v < p.V then
      a.value (l.head*p.valueHead+(l.chunk*p.BT+j.val)*p.valueTime+v) else 0)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO
