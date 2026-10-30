import bench.bug_detection.tasks.ben_silu_x_sigmoid.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_silu_x_sigmoid
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical


def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => xs.x r j / (1 + Real.exp (-xs.x r j))

end Contract
end
end VeriTile.Bench.BugDetection.ben_silu_x_sigmoid
