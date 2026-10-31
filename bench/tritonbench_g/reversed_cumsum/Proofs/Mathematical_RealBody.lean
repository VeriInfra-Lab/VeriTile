import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealProgram
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Erasure

/-! Exact whole-source bridge for the signed mirrored loop. Source controls
remain present, including the original post-store nonnegative-index branch. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def reverseStart (p : Layout) : Op .int [] :=
  .sub .int .nil (.constInt 0)
    (.sub .int .nil (.castNatToInt (stopOp p)) (.constInt 1))

end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
