import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHAccessSafety
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
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

structure GateBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : gateK = Bool.true → KeyBounds p l r.gate c bounds
  keyEnd : gateK = Bool.true → EndpointBounds p l r.gate c p.keyHead p.keyTime p.keyFeature p.BK l.key bounds
  value : gateK = Bool.false → ValueBounds p l r.gate c bounds
  valueEnd : gateK = Bool.false → EndpointBounds p l r.gate c p.valueHead p.valueTime p.valueFeature p.BV l.value bounds

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (gateK : Bool)
    (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.key c bounds
  value : ValueBounds p l r.value c bounds
  history : HistoryBounds p l r.history c bounds
  gate : GateBounds p l r c gateK bounds

def LoopBounds (p : Layout) (l : Launch) (r : Regions) (gateK : Bool) (bounds : RegionBounds) : Prop :=
  ∀ c, c < p.NT → StepBounds p l r c gateK bounds

structure Bounds (p : Layout) (l : Launch) (r : Regions) (gateK hasSeed storeFinal : Bool)
    (bounds : RegionBounds) : Prop where
  initial : hasSeed = Bool.true → StateBounds p l r.seed bounds
  loop : LoopBounds p l r gateK bounds
  final : storeFinal = Bool.true → StateBounds p l r.final bounds

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Bounds
