import bench.tritonbench_g.geglu_tanh_triton.Proofs.Mathematical_Math
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Math
noncomputable section
namespace Original
abbrev scale := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.scale
abbrev argument := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.argument
abbrev gate := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.gate
abbrev forward := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.forward
abbrev gateFactor := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.gateFactor
abbrev backward := VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.backward
end Original
open Original

/-- The hyperbolic derivative is established analytically, independently of
the written backward program and with no floating-point premise. -/
theorem tanh_derivative (x : ℝ) :
    HasDerivAt Real.tanh (1 - Real.tanh x ^ 2) x := by
  have hn : Real.cosh x ≠ 0 := ne_of_gt (Real.cosh_pos x)
  have h := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hn
  have scalar : (Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) /
      Real.cosh x ^ 2 = 1 - Real.tanh x ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh]
    field_simp
    <;> ring
  have functions : (Real.sinh / Real.cosh) = Real.tanh := by
    funext y
    exact (Real.tanh_eq_sinh_div_cosh y).symm
  rw [functions, scalar] at h
  exact h

theorem argument_derivative (a : ℝ) :
    HasDerivAt argument (scale * (1 + 3 * 0.044715 * a ^ 2)) a := by
  have h := ((hasDerivAt_id a).add
    (((hasDerivAt_id a).pow 3).const_mul (0.044715 : ℝ))).const_mul scale
  convert h using 1 <;>
    dsimp [argument, VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.argument] <;>
    norm_num only [Nat.cast_ofNat] <;> ring

/-- True derivative of the independent exact-decimal GELU approximation.
The source decimal alpha is not replaced by the ideal sqrt(2/pi). -/
theorem gate_derivative (a : ℝ) : HasDerivAt gate (gateFactor a) a := by
  have h := ((hasDerivAt_id a).div_const (2 : ℝ)).mul
    (((tanh_derivative (argument a)).comp a (argument_derivative a)).const_add 1)
  convert h using 1 <;>
    dsimp [gateFactor, VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.gateFactor] <;>
    ring

theorem gate_deriv (a : ℝ) : deriv gate a = gateFactor a :=
  (gate_derivative a).deriv

theorem forward_gate_derivative (a b : ℝ) :
    HasDerivAt (fun x => forward x b) (b * gateFactor a) a := by
  simpa only [forward, VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.forward,
    mul_comm] using (gate_derivative a).mul_const b

theorem forward_value_derivative (a b : ℝ) :
    HasDerivAt (fun x => forward a x) (gate a) b := by
  simpa only [forward, VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.forward,
    mul_one] using (hasDerivAt_id b).const_mul (gate a)

/-- Both returned gradients are upstream times the actual two partials. -/
theorem backward_is_gradient (dc a b : ℝ) :
    backward dc a b =
      (dc * deriv (fun x => forward x b) a,
       dc * deriv (fun x => forward a x) b) := by
  rw [(forward_gate_derivative a b).deriv, (forward_value_derivative a b).deriv]
  simp only [backward, VeriTile.Bench.TritonBenchG.GegluTanhTriton.Math.backward]
  congr 1
  ring

#axiomsClean backward_is_gradient
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Math
