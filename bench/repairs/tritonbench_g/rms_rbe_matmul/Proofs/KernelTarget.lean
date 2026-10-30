import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelMath
noncomputable section

/-- Independent logical inputs. `rms` is the original per-element
Tensor.to(weight.dtype) interpretation, followed by E5M2 decoding for the
byte arm. It is an input scalar conversion, never an output premise. -/
structure Input where
  x : Nat → ℝ
  rms : Nat → ℝ
  weight : Nat → Nat → ℝ

def epsilon : ℝ := 1 / 1000000
def theta : ℝ := 10000
def squareMean (K : Nat) (v : Input) : ℝ :=
  (∑ k ∈ Finset.range K, v.x k ^ 2) / (K : ℝ)
def inverse (K : Nat) (v : Input) : ℝ :=
  1 / Real.sqrt (squareMean K v + epsilon)
def projection (K : Nat) (v : Input) (column : Nat) : ℝ :=
  (∑ k ∈ Finset.range K, v.x k * v.rms k * v.weight column k) * inverse K v
def angle (headDim : Nat) (start : Int) (row column : Nat) : ℝ :=
  ((start : ℝ) + (row : ℝ)) /
    Real.rpow theta (((column - column % 2) % headDim : Nat) / (headDim : ℝ))

/-- RMS-normalized single projection, followed optionally by adjacent-pair
rotation whose frequencies restart at each caller-selected head boundary.
No program state, source graph, compiler plan or memory write occurs here. -/
def output (K headDim : Nat) (start : Int) (rotary : Bool)
    (v : Input) (row column : Nat) : ℝ :=
  if rotary then
    let even := column - column % 2
    let phi := angle headDim start row column
    if column % 2 = 0 then
      projection K v even * Real.cos phi - projection K v (even + 1) * Real.sin phi
    else
      projection K v even * Real.sin phi + projection K v (even + 1) * Real.cos phi
  else projection K v column

theorem squareMean_nonnegative (K : Nat) (v : Input) : 0 ≤ squareMean K v := by
  apply div_nonneg
  · exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  · exact Nat.cast_nonneg K

theorem inverse_positive (K : Nat) (v : Input) : 0 < inverse K v := by
  apply one_div_pos.mpr
  apply Real.sqrt_pos.2
  have h := squareMean_nonnegative K v
  norm_num [epsilon] at *
  linarith

#axiomsClean squareMean_nonnegative
#axiomsClean inverse_positive
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelMath
