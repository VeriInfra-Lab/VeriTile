import bench.bug_detection.tasks.swiglu__default.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.swiglu__default
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def silu (x : ℝ) : ℝ := x / (1 + Real.exp (-x))
def gateValue (xs : IO.Inputs) (r : Fin 16) (j : Fin 6144) : ℝ := ∑ h, xs.x r h * xs.gate j h
def upValue (xs : IO.Inputs) (r : Fin 16) (j : Fin 6144) : ℝ := ∑ h, xs.x r h * xs.up j h
def activation : ℝ → ℝ := silu

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,h) => ∑ j, (activation (gateValue xs r j) * upValue xs r j) * xs.down h j

end Contract
end
end VeriTile.Bench.BugDetection.swiglu__default
