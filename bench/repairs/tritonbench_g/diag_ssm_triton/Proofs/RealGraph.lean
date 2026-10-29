import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ScalarGraphs
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Adjoint

/-! Independent Real interpretation of the shared scalar graphs. Native
rounding is not erased in any FP theorem; this is the separate mathematical
projection, connected to the unchanged complete original typed bodies. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.RealGraph
open Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section

def arithmetic : Operations ℝ := ⟨0, (· + ·), (· - ·), (· * ·), fun a b c => a * b + c, Neg.neg⟩
def complex (a : Pair ℝ) : ℂ := ⟨a.re, a.im⟩
def components (a : ℂ) : Pair ℝ := ⟨a.re, a.im⟩

@[simp] theorem complex_components (a : ℂ) : complex (components a) = a := rfl
@[simp] theorem components_complex (a : Pair ℝ) : components (complex a) = a := by cases a; rfl

theorem real_forward (graph : Graph) (s transition x : ℝ) :
    realForward graph arithmetic s transition x = s * transition + x := by
  cases graph <;> rfl

theorem complex_product (graph : Graph) (a b : Pair ℝ) :
    complex (complexProduct graph arithmetic a b) = complex a * complex b := by
  cases graph <;> apply Complex.ext <;>
    simp [complexProduct, arithmetic, complex, Complex.mul_re, Complex.mul_im] <;> ring

theorem complex_forward (graph : Graph) (s transition x : Pair ℝ) :
    complex (complexForward graph arithmetic s transition x) = complex s * complex transition + complex x := by
  change complex (complexProduct graph arithmetic s transition) + complex x = _
  rw [complex_product]

theorem gradient_conjugate (g : Pair ℝ) :
    complex (gradientConjugate arithmetic g) = starRingEnd ℂ (complex g) := by
  apply Complex.ext <;> simp [complex, gradientConjugate, arithmetic]

theorem complex_backward (graph : Graph) (previous transition gy : Pair ℝ) (r : BackwardRegisters ℝ) :
    let next := complexBackward graph arithmetic previous transition gy r
    complex next.inputGradient = complex gy + starRingEnd ℂ (complex r.carry) ∧
    complex (gradientConjugate arithmetic next.registers.carry) =
      (complex gy + starRingEnd ℂ (complex r.carry)) * starRingEnd ℂ (complex transition) ∧
    complex (gradientConjugate arithmetic next.registers.transition) =
      starRingEnd ℂ (complex r.transition) +
        (complex gy + starRingEnd ℂ (complex r.carry)) * starRingEnd ℂ (complex previous) := by
  cases graph <;>
    simp only [complexBackward, accumulatedGradient, gradientConjugate, complexProduct, arithmetic, complex]
  all_goals constructor
  all_goals try constructor
  all_goals apply Complex.ext <;> simp [Complex.mul_re, Complex.mul_im] <;> ring

/-- The independent forward recurrence used by the complete original source
proof has the same state update as the Real graph; its closed form is reused. -/
theorem forward_closed_form (initial transition : ℂ) (input : Nat → ℂ) (t : Nat) :
    Reference.forwardState initial transition input t = initial * transition ^ t +
      ∑ k ∈ Finset.range t, input k * transition ^ (t - 1 - k) :=
  Reference.forward_closed_form initial transition input t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.RealGraph
