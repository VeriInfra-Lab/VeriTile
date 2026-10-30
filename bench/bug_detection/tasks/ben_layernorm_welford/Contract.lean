import bench.bug_detection.tasks.ben_layernorm_welford.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_layernorm_welford
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def mean (xs : IO.Inputs) (r : Fin 16) : ℝ := (∑ j, xs.x r j) / 2048
def variance (xs : IO.Inputs) (r : Fin 16) : ℝ := (∑ j, ((xs.x r j - mean xs r))^2) / 2048

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => xs.weight j * (xs.x r j - mean xs r) / Real.sqrt ((∑ k : Fin 2048, (xs.x r k - mean xs r)^2) / 2048 + 1/1000000) + xs.bias j

end Contract
end
end VeriTile.Bench.BugDetection.ben_layernorm_welford
