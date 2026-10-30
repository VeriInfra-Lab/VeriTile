import bench.tritonbench_g.lightning_attention.Grid
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section

/-- The actual masked addresses of a forward program. No width divisibility
or layout assumption is smuggled into this generic source-safety lemma. -/
structure ForwardBounds (p : Parameters) (pid : PID) (bounds : RegionName → Nat) : Prop where
  query : ∀ t a, t < p.N → a < p.D → pid.head*p.N*p.D+t*p.D+a < bounds "query"
  key : ∀ t a, t < p.N → a < p.D → pid.head*p.N*p.D+t*p.D+a < bounds "key"
  value : ∀ t c, t < p.N → c < p.BM →
    pid.head*p.N*p.E+pid.channel*p.BM+t*p.E+c < bounds "value"
  output : ∀ t c, t < p.N → c < p.BM →
    pid.head*p.N*p.E+pid.channel*p.BM+t*p.E+c < bounds "output"

/-- Dense reads and gradient updates for one head; row masks are evaluated by
the original source, including both descending scans. -/
structure HeadBounds (p : Parameters) (head : Nat) (bounds : RegionName → Nat) : Prop where
  query : ∀ t a, t < p.N → a < p.D → head*p.N*p.D+t*p.D+a < bounds "query"
  key : ∀ t a, t < p.N → a < p.D → head*p.N*p.D+t*p.D+a < bounds "key"
  value : ∀ t c, t < p.N → c < p.E → head*p.N*p.E+t*p.E+c < bounds "value"
  incoming : ∀ t c, t < p.N → c < p.E → head*p.N*p.E+t*p.E+c < bounds "incoming"
  queryGradient : ∀ t a, t < p.N → a < p.D → head*p.N*p.D+t*p.D+a < bounds "queryGradient"
  keyGradient : ∀ t a, t < p.N → a < p.D → head*p.N*p.D+t*p.D+a < bounds "keyGradient"
  valueGradient : ∀ t c, t < p.N → c < p.E → head*p.N*p.E+t*p.E+c < bounds "valueGradient"

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
