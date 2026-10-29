import bench.bug_detection.tasks.ben_col_parallel_tp2.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_col_parallel_tp2
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical


def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => (∑ k : Fin 2048, xs.x r k * xs.weight j k)

end Contract
end
end VeriTile.Bench.BugDetection.ben_col_parallel_tp2
