import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedExecution
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHAccessSafety

/-! Allocation bounds for actual local-parent pointers. The generic checked
load/store safety lemmas are reused unchanged. No oracle values are assumed. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedAccessSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram RealHAccessSafety

def KeyBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.keyHead+l.key*p.BK*p.keyFeature+c*p.BT*p.keyTime)
    p.BK p.BT p.BK p.BT p.keyFeature p.keyTime 0 0

def ValueBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.valueHead+c*p.BT*p.valueTime+l.value*p.BV*p.valueFeature)
    p.BT p.BV p.BT p.BV p.valueTime p.valueFeature 0 0

def StateBounds (p : Layout) (l : Launch) (r : RegionName) :=
  MatrixBounds r (l.head*p.K*p.V+l.key*p.BK*p.V+l.value*p.BV)
    p.BK p.BV p.BK p.BV p.V 1 0 0

def HistoryBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.historyHead+c*p.K*p.V+l.key*p.BK*p.historyKey+l.value*p.BV*p.historyValue)
    p.BK p.BV p.BK p.BV p.historyKey p.historyValue 0 0

def EndpointBounds (p : Layout) (l : Launch) (r : RegionName)
    (c head time stride block index : Nat) :=
  VectorBounds r (l.head*head+(c*p.BT+p.BT-1)*time+index*block*stride)
    block block stride 0

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedAccessSafety
