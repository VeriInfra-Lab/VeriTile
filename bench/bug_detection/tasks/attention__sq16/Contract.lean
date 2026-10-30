import bench.bug_detection.tasks.attention__sq16.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.attention__sq16
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def kv (h : Fin 16) : Fin 8 := ⟨h.val / 2, by omega⟩
abbrev q := IO.Inputs.q
abbrev k := IO.Inputs.k
def score (xs : IO.Inputs) (h : Fin 16) (r j : Fin 16) : ℝ :=
  (∑ d, q xs h r d * k xs (kv h) j d) / Real.sqrt (128 * 1)
def blocked (r j : Fin 16) : Prop := j.val > r.val
def weight (xs : IO.Inputs) (h : Fin 16) (r j : Fin 16) : ℝ := Real.exp (score xs h r j + if blocked r j then -338953138925153547590470800371487866880 else 0)

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,h,d) => ∑ j : Fin 16, (weight xs h r j / (∑ k : Fin 16, weight xs h r k)) * xs.v (kv h) j d

end Contract
end
end VeriTile.Bench.BugDetection.attention__sq16
