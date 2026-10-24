import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.PrimitiveNumerics

namespace VeriTile.Bench.Repairs.DiagSsmTriton.GraphNumerics
open Kernels PrimitiveNumerics
set_option autoImplicit false
set_option maxHeartbeats 3000000
noncomputable section

def registerLeaves (r : BackwardRegisters Word) : BackwardRegisters Expr := ⟨Leaves r.carry, Leaves r.transition⟩

def backwardExpr (route : Route) (graph : Graph) (previous transition gy : Pair Word)
    (registers : BackwardRegisters Word) : BackwardStep Expr :=
  Kernels.backward route graph expressionOps (Leaves previous) (Leaves transition) (Leaves gy) (registerLeaves registers)

def effective (route : Route) (pair : Pair Word) : ℂ := observe route (project route (realPair pair))

theorem observe_add (route : Route) (a b : ℂ) : observe route (a + b) = observe route a + observe route b := by
  cases route <;> simp [observe]

theorem eval_backward (ops : Operations Word) (route : Route) (graph : Graph)
    (previous transition gy : Pair Word) (registers : BackwardRegisters Word) :
    let expr := backwardExpr route graph previous transition gy registers
    let actual := Kernels.backward route graph ops previous transition gy registers
    evalPair ops expr.inputGradient = actual.inputGradient ∧
      evalPair ops expr.registers.carry = actual.registers.carry ∧
      evalPair ops expr.registers.transition = actual.registers.transition := by
  cases route <;> cases graph <;> exact ⟨rfl, rfl, rfl⟩

theorem ideal_backward (route : Route) (graph : Graph) (previous transition gy : Pair Word)
    (registers : BackwardRegisters Word) :
    let expr := backwardExpr route graph previous transition gy registers
    let a := project route (realPair gy) + effective route registers.carry
    project route (idealPair expr.inputGradient) = a ∧
      observe route (project route (idealPair expr.registers.carry)) = a * starRingEnd ℂ (project route (realPair transition)) ∧
      observe route (project route (idealPair expr.registers.transition)) =
        effective route registers.transition + a * starRingEnd ℂ (project route (realPair previous)) := by
  cases route <;> cases graph
  all_goals dsimp only
  all_goals constructor
  all_goals try constructor
  all_goals apply Complex.ext <;>
    simp [backwardExpr, Kernels.backward, realBackward, complexBackward, accumulatedGradient, complexProduct,
      gradientConjugate, registerLeaves, expressionOps, Leaves, idealPair, Expr.ideal, project, realPair,
      effective, observe, Complex.mul_re, Complex.mul_im] <;> try ring
  all_goals simp

theorem backward_error (ops : Operations Word) (route : Route) (graph : Graph)
    (previous transition gy : Pair Word) (registers : BackwardRegisters Word) :
    let expr := backwardExpr route graph previous transition gy registers
    let actual := Kernels.backward route graph ops previous transition gy registers
    let a := project route (realPair gy) + effective route registers.carry
    project route (realPair actual.inputGradient) = a + project route (errorPair ops expr.inputGradient) ∧
      effective route actual.registers.carry =
        a * starRingEnd ℂ (project route (realPair transition)) +
          observe route (project route (errorPair ops expr.registers.carry)) ∧
      effective route actual.registers.transition =
        effective route registers.transition + a * starRingEnd ℂ (project route (realPair previous)) +
          observe route (project route (errorPair ops expr.registers.transition)) := by
  dsimp only
  have eval := eval_backward ops route graph previous transition gy registers
  have ideal := ideal_backward route graph previous transition gy registers
  constructor
  · rw [← eval.1, pair_error, ideal.1]
  constructor
  · unfold effective
    rw [← eval.2.1, pair_error, observe_add, ideal.2.1]
    rfl
  · unfold effective
    rw [← eval.2.2, pair_error, observe_add, ideal.2.2]
    rfl

def terminalExpr (route : Route) (pair : Pair Word) : Pair Expr := terminalGradient route expressionOps (Leaves pair)

theorem terminal_error (ops : Operations Word) (route : Route) (pair : Pair Word) :
    project route (realPair (terminalGradient route ops pair)) =
      effective route pair + project route (errorPair ops (terminalExpr route pair)) := by
  have eval : evalPair ops (terminalExpr route pair) = terminalGradient route ops pair := by cases route <;> rfl
  have ideal : project route (idealPair (terminalExpr route pair)) = effective route pair := by
    cases route <;> apply Complex.ext <;>
      simp [project, idealPair, terminalExpr, terminalGradient, gradientConjugate,
        expressionOps, Leaves, Expr.ideal, effective, observe, realPair]
  rw [← eval, pair_error, ideal]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.GraphNumerics
