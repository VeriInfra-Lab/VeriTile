import bench.tritonbench_g.attention_score.Kernels
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.SourcePair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Reused Geometry
noncomputable section

/-- The existing scalar-left comparison's local domain checks only actual
reached FP32 qk lanes and scale; unrestricted native IO remains auxiliary. -/
def requirements (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : List Site :=
  match Source.intRange? (kernel c offset size scale window complement).surfaceBody "start_m" with
  | none => []
  | some loop => [.intRange loop.before "start_m" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) c.B [c.B]
        (.ref .real [c.B,c.B] "qk") (.ref .real [] "qk_scale"))]

def original (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : Program where
  kernel := kernel c offset size scale window complement
  domain := requirements c offset size scale window complement

def scalarLeft (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) : Program :=
  { original c offset size scale window complement with
    kernel := scalarLeftKernel c offset size scale window complement }

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.SourcePair
