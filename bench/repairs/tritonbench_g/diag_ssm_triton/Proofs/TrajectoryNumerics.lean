import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.GraphNumerics
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardGrid
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.NumericalPropagation

/-! Primitive residuals propagated along the actual source registers. Forward
error has a closed form. Backward error reads the actual saved y bytes and
retains their discrepancy from the independent ideal trajectory explicitly. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.TrajectoryNumerics
open Kernels ByteMemory Execution PrimitiveNumerics GraphNumerics
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
set_option autoImplicit false
noncomputable section

def forwardTransition (codec : Codec Word) (route : Route) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) : Pair Word :=
  readTile codec route ops views.transition d.width pid (transitionIndex d pid) m j

def forwardInput (codec : Codec Word) (route : Route) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) : Pair Word :=
  readTile codec route ops views.input d.width pid (timeIndex d t pid) m j

def forwardResidual (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) : ℂ :=
  project route (errorPair ops (forwardExpr route graph
    (forwardValues codec route graph ops d views pid m t j)
    (forwardTransition codec route ops d views pid m j) (forwardInput codec route ops d views pid m j t)))

def idealForward (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) : ℂ :=
  Reference.forwardState
    (project route (realPair (forwardValues codec route graph ops d views pid m 0 j)))
    (project route (realPair (forwardTransition codec route ops d views pid m j)))
    (fun t => project route (realPair (forwardInput codec route ops d views pid m j t))) t

theorem forward_error_recurrence (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) :
    project route (realPair (forwardValues codec route graph ops d views pid m (t + 1) j)) -
        idealForward codec route graph ops d views pid m j (t + 1) =
      (project route (realPair (forwardValues codec route graph ops d views pid m t j)) -
        idealForward codec route graph ops d views pid m j t) *
          project route (realPair (forwardTransition codec route ops d views pid m j)) +
        forwardResidual codec route graph ops d views pid m j t := by
  apply NumericalPropagation.forward_error_step
  · rfl
  · exact PrimitiveNumerics.forward_error ops route graph _ _ _

/-- Exact accumulated residual, with no unproved global kernel-error bound. -/
theorem forward_error_sum (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) :
    project route (realPair (forwardValues codec route graph ops d views pid m t j)) -
        idealForward codec route graph ops d views pid m j t =
      ∑ k ∈ Finset.range t, forwardResidual codec route graph ops d views pid m j k *
        project route (realPair (forwardTransition codec route ops d views pid m j)) ^ (t - 1 - k) := by
  let transition := project route (realPair (forwardTransition codec route ops d views pid m j))
  let residual := forwardResidual codec route graph ops d views pid m j
  have recurrence : ∀ t,
      project route (realPair (forwardValues codec route graph ops d views pid m t j)) -
        idealForward codec route graph ops d views pid m j t = Reference.forwardState 0 transition residual t := by
    intro t
    induction t with
    | zero => simp [idealForward, Reference.forwardState]
    | succ t ih =>
      rw [forward_error_recurrence, ih]
      rfl
  rw [recurrence, Reference.forward_closed_form]
  simp [transition, residual]

def backwardPrevious (codec : Codec Word) (route : Route) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) (i : Nat) : Pair Word :=
  readTile codec route ops (previousView d views i) d.width pid (previousIndex d pid i) m j

def backwardTransition (codec : Codec Word) (route : Route) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) : Pair Word :=
  readTile codec route ops views.transition d.width pid (transitionIndex d pid) m j

def cotangent (codec : Codec Word) (route : Route) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) (t : Nat) : Pair Word :=
  readTile codec route ops views.gradOutput d.width pid (timeIndex d t pid) m j

def backwardExpression (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) (i : Nat) : BackwardStep Expr :=
  backwardExpr route graph (backwardPrevious codec route ops d views pid m j i)
    (backwardTransition codec route ops d views pid m j)
    (cotangent codec route ops d views pid m j (d.length - 1 - i))
    (backwardValues codec route graph ops d views pid m i j)

/-- This is the full local reverse-step equation, derived from the actual
scalar expression graph. The residuals are sums/products of local primitive
residuals; they are not free whole-step assumptions. -/
theorem backward_step (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) (i : Nat) :
    let r := backwardValues codec route graph ops d views pid m i j
    let next := backwardValues codec route graph ops d views pid m (i + 1) j
    let expr := backwardExpression codec route graph ops d views pid m j i
    let a := project route (realPair (cotangent codec route ops d views pid m j (d.length - 1 - i))) + effective route r.carry
    project route (realPair (backwardInputValues codec route graph ops d views pid m i j)) =
      a + project route (errorPair ops expr.inputGradient) ∧
      effective route next.carry = a * starRingEnd ℂ (project route (realPair (backwardTransition codec route ops d views pid m j))) +
        observe route (project route (errorPair ops expr.registers.carry)) ∧
      effective route next.transition = effective route r.transition +
        a * starRingEnd ℂ (project route (realPair (backwardPrevious codec route ops d views pid m j i))) +
          observe route (project route (errorPair ops expr.registers.transition)) :=
  GraphNumerics.backward_error ops route graph _ _ _ _

/-- The saved-state error term is retained even when current carry arithmetic
would be exact. `idealState` is the independent forward trajectory; the actual
previous state is loaded from the saved output byte window. -/
theorem transition_error_step (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (j : Lane) (i : Nat)
    (idealState : Nat → ℂ) :
    let transition := project route (realPair (backwardTransition codec route ops d views pid m j))
    let gy := fun t => project route (realPair (cotangent codec route ops d views pid m j t))
    let r := backwardValues codec route graph ops d views pid m i j
    let actualA := gy (d.length - 1 - i) + effective route r.carry
    let idealA := Adjoint.inputGradient d.length transition gy i
    let actualPrevious := project route (realPair (backwardPrevious codec route ops d views pid m j i))
    effective route (backwardValues codec route graph ops d views pid m (i + 1) j).transition -
        Adjoint.transitionGradient d.length transition idealState gy (i + 1) =
      (effective route r.transition - Adjoint.transitionGradient d.length transition idealState gy i) +
      (actualA - idealA) * starRingEnd ℂ (idealState (d.length - 1 - i)) +
      actualA * starRingEnd ℂ (actualPrevious - idealState (d.length - 1 - i)) +
      observe route (project route (errorPair ops (backwardExpression codec route graph ops d views pid m j i).registers.transition)) := by
  dsimp only
  rw [(backward_step codec route graph ops d views pid m j i).2.2, Adjoint.transitionGradient]
  exact NumericalPropagation.reverse_transition_error _ _ _ _ _ _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.TrajectoryNumerics
