import bench.bug_detection.tasks.attn_fa_bf16_accum.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.attn_fa_bf16_accum
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def kv (h : Fin 1) : Fin 1 := ⟨h.val / 1, by omega⟩
abbrev q := IO.Inputs.q
abbrev k := IO.Inputs.k
def score (xs : IO.Inputs) (h : Fin 1) (r j : Fin 16384) : ℝ :=
  (∑ d, q xs h r d * k xs (kv h) j d) / Real.sqrt (128 * 1)
def blocked (r j : Fin 16384) : Prop := j.val > r.val
def weight (xs : IO.Inputs) (h : Fin 1) (r j : Fin 16384) : ℝ := if blocked r j then 0 else Real.exp (score xs h r j)

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,h,d) => ∑ j : Fin 16384, (weight xs h r j / (∑ k : Fin 16384, weight xs h r k)) * xs.v (kv h) j d

end Contract
end
end VeriTile.Bench.BugDetection.attn_fa_bf16_accum
