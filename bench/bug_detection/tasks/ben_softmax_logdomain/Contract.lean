import bench.bug_detection.tasks.ben_softmax_logdomain.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.ben_softmax_logdomain
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical


def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,j) => Real.exp (xs.x r j) / ∑ k : Fin 256, Real.exp (xs.x r k)

end Contract
end
end VeriTile.Bench.BugDetection.ben_softmax_logdomain
