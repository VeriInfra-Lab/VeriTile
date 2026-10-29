import bench.bug_detection.tasks.rmsnorm_no_upcast.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.rmsnorm_no_upcast
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def variance (xs : IO.Inputs) (r : Fin 16) : ℝ := (∑ j, (xs.x r j)^2) / 2048

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => xs.weight j * xs.x r j / Real.sqrt ((∑ k : Fin 2048, (xs.x r k)^2) / 2048 + 1/1000000)

end Contract
end
end VeriTile.Bench.BugDetection.rmsnorm_no_upcast
