import bench.bug_detection.tasks.moe_capacity_drop.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.BugDetection.moe_capacity_drop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Contract
open scoped BigOperators Classical

def silu (x : ℝ) : ℝ := x / (1 + Real.exp (-x))
def score (xs : IO.Inputs) (t : Fin 16) (e : Fin 128) : ℝ :=
  1 / (1 + Real.exp (-(∑ h, xs.x t h * xs.router e h)))
/-- Descending scores, breaking equal scores by increasing expert index.
The FP backend binding must establish this tie policy or restrict its domain. -/
def ranked (xs : IO.Inputs) (t : Fin 16) : List (Fin 128) :=
  (List.finRange 128).mergeSort (fun a b => decide (score xs t a > score xs t b ∨ (score xs t a = score xs t b ∧ a.val ≤ b.val)))
def route (xs : IO.Inputs) (t : Fin 16) (j : Fin 6) : Fin 128 :=
  (ranked xs t)[j.val]?.getD 0
def probability (xs : IO.Inputs) (t : Fin 16) (j : Fin 6) : ℝ :=
  score xs t (route xs t j) / ((∑ k : Fin 6, score xs t (route xs t k)) + 1/100000000000000000000) * (2826/1000)
def expertValue (xs : IO.Inputs) (t : Fin 16) (e : Fin 128) (h : Fin 2048) : ℝ :=
  ∑ j : Fin 640, (silu (∑ k, xs.x t k * xs.gate e j k) * (∑ k, xs.x t k * xs.up e j k)) * xs.down e h j
def sharedValue (xs : IO.Inputs) (t : Fin 16) (h : Fin 2048) : ℝ :=
  ∑ j : Fin 512, (silu (∑ k, xs.x t k * xs.shared_gate j k) * (∑ k, xs.x t k * xs.shared_up j k)) * xs.shared_down h j

def Allowed (_xs : IO.Inputs) : Prop := True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (t,h) => (∑ j : Fin 6, probability xs t j * expertValue xs t (route xs t j) h) + sharedValue xs t h

end Contract
end
end VeriTile.Bench.BugDetection.moe_capacity_drop
