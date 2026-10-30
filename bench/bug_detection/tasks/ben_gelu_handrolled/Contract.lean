import bench.bug_detection.tasks.ben_gelu_handrolled.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_gelu_handrolled
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical


def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => 0.5 * xs.x r j * (1 + Real.tanh (0.7978845608028654 * (xs.x r j + 0.044715 * (xs.x r j * xs.x r j * xs.x r j))))

end Contract
end
end VeriTile.Bench.BugDetection.ben_gelu_handrolled
