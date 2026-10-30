import bench.bug_detection.tasks.attn_qknorm_wrong_dim.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.attn_qknorm_wrong_dim
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def kv (h : Fin 16) : Fin 16 := ⟨h.val / 1, by omega⟩
def q (xs : IO.Inputs) (h : Fin 16) (r : Fin 16) (d : Fin 128) : ℝ := xs.q h r d * xs.qw d / Real.sqrt ((∑ j, (xs.q h r j)^2) / 128 + 1/1000000)
def k (xs : IO.Inputs) (h : Fin 16) (r : Fin 16) (d : Fin 128) : ℝ := xs.k h r d * xs.kw d / Real.sqrt ((∑ j, (xs.k h r j)^2) / 128 + 1/1000000)
def score (xs : IO.Inputs) (h : Fin 16) (r j : Fin 16) : ℝ :=
  (∑ d, q xs h r d * k xs (kv h) j d) / Real.sqrt (128 * 1)
def blocked (r j : Fin 16) : Prop := j.val > r.val
def weight (xs : IO.Inputs) (h : Fin 16) (r j : Fin 16) : ℝ := if blocked r j then 0 else Real.exp (score xs h r j)

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (r,h,d) => ∑ j : Fin 16, (weight xs h r j / (∑ k : Fin 16, weight xs h r k)) * xs.v (kv h) j d

end Contract
end
end VeriTile.Bench.BugDetection.attn_qknorm_wrong_dim
