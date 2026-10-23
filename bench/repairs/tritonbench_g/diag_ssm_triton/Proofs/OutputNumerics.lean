import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardWrapper
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.TrajectoryNumerics

/-! Numerical equations for actual returned storage, not unattached scan
recurrences. Every residual expands to actual local primitive operands. Finite
numerical domains and native primitive bounds are stated separately; the Word
execution and caller byte frames cover all bit patterns. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.OutputNumerics
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths Dispatch
open PrimitiveNumerics GraphNumerics TrajectoryNumerics
set_option autoImplicit false
noncomputable section

def storedPair (codec : Codec Word) (route : Route) (ops : Operations Word) (view : View) (index : Nat) (m : Heap) : Pair Word :=
  ⟨load codec m (view.address (route.components * index)),
    match route with
    | .real32 => ops.zero
    | .complex64 => load codec m (view.address (route.components * index + 1))⟩

theorem stored_projection (codec : Codec Word) (route : Route) (ops : Operations Word)
    (view : View) (index : Nat) (m : Heap) (values : Pair Word)
    (readback : ∀ k, k ∈ componentList route →
      load codec m (view.address (route.components * index + k)) = getComponent values k) :
    project route (realPair (storedPair codec route ops view index m)) = project route (realPair values) := by
  have re := readback 0 (by cases route <;> simp [componentList])
  simp only [Nat.add_zero, getComponent, if_pos rfl] at re
  cases route
  · simp [project, realPair, storedPair, re]
  · have im := readback 1 (by simp [componentList])
    simp only [getComponent, if_neg (by decide : (1 : Nat) ≠ 0)] at im
    simp [project, realPair, storedPair, re, im]

theorem forward_returned_error (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (report : ForwardWrapper.Report codec graph ops api choices route d s x transition returned before after)
    (t : Nat) (ht : t < d.length) (pid : Nat) (j : Lane) (hj : active d.width pid j) :
    let views := forwardViews report.path.initial report.path.input report.path.parameter returned
    project route (realPair (storedPair codec route ops returned.view (timeIndex d t pid j) after.heap)) -
        idealForward codec route graph ops d views pid report.dispatch.start.heap j (t + 1) =
      ∑ k ∈ Finset.range (t + 1), forwardResidual codec route graph ops d views pid report.dispatch.start.heap j k *
        project route (realPair (forwardTransition codec route ops d views pid report.dispatch.start.heap j)) ^ (t - k) := by
  dsimp only
  rw [stored_projection codec route ops returned.view _ after.heap _ (report.readback t ht pid j hj)]
  simpa using forward_error_sum codec route graph ops d
    (forwardViews report.path.initial report.path.input report.path.parameter returned) pid report.dispatch.start.heap j (t + 1)

theorem input_returned_error (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (i : Nat) (hi : i < d.length) (pid : Nat) (j : Lane) (hj : active d.width pid j) :
    let views := backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
      report.path.contiguousGradient report.path.gs report.path.gx report.path.gl
    let m := report.dispatch.start.heap
    let transition := project route (realPair (backwardTransition codec route ops d views pid m j))
    let cotangents := fun t => project route (realPair (cotangent codec route ops d views pid m j t))
    project route (realPair (storedPair codec route ops returned.2.1.view (timeIndex d (d.length - 1 - i) pid j) after.heap)) -
        Adjoint.inputGradient d.length transition cotangents i =
      effective route (backwardValues codec route graph ops d views pid m i j).carry -
        Adjoint.carry d.length transition cotangents i +
      project route (errorPair ops (backwardExpression codec route graph ops d views pid m j i).inputGradient) := by
  dsimp only
  rw [stored_projection codec route ops returned.2.1.view _ after.heap _ (report.inputReadback i hi pid j hj)]
  rw [(backward_step codec route graph ops d _ pid report.dispatch.start.heap j i).1]
  unfold Adjoint.inputGradient
  ring

theorem initial_returned_error (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (pid : Nat) (j : Lane) (hj : active d.width pid j) (ideal : ℂ) :
    let views := backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
      report.path.contiguousGradient report.path.gs report.path.gx report.path.gl
    let carry := (backwardValues codec route graph ops d views pid report.dispatch.start.heap d.length j).carry
    project route (realPair (storedPair codec route ops returned.1.view (columnIndex pid j) after.heap)) - ideal =
      (effective route carry - ideal) + project route (errorPair ops (terminalExpr route carry)) := by
  dsimp only
  rw [stored_projection codec route ops returned.1.view _ after.heap _ (report.initialReadback pid j hj), terminal_error]
  ring

def sumExpr (tree : Reduction.Tree) (values : Nat → Word) : Expr :=
  match tree with
  | .zero => .zero
  | .leaf b => .leaf (values b)
  | .add left right => .add (sumExpr left values) (sumExpr right values)

theorem sumExpr_evaluate (ops : Operations Word) (tree : Reduction.Tree) (values : Nat → Word) :
    (sumExpr tree values).evaluate ops = tree.evaluate ops.zero ops.add values := by
  induction tree <;> simp [sumExpr, Expr.evaluate, Reduction.Tree.evaluate, *]

theorem sumExpr_ideal (tree : Reduction.Tree) (values : Nat → Word) :
    (sumExpr tree values).ideal = (tree.leaves.map (fun b => real (values b))).sum := by
  induction tree <;> simp [sumExpr, Expr.ideal, Reduction.Tree.leaves, *]

theorem sumExpr_plan (batch dim components : Nat) (plan : Reduction.Plan batch dim components)
    (q : Nat) (hq : q < components * dim) (values : Nat → Word) :
    (sumExpr (plan.tree q) values).ideal = (Finset.range batch).sum (fun b => real (values b)) := by
  rw [sumExpr_ideal, (List.Perm.map (fun b => real (values b)) (plan.leaves q hq)).sum_eq]
  have rangeSum (n : Nat) : ((List.range n).map (fun b => real (values b))).sum =
      (Finset.range n).sum (fun b => real (values b)) := by
    induction n with
    | zero => simp
    | succ n ih => simp [List.range_succ, Finset.sum_range_succ, ih]
  exact rangeSum batch

/-- Final reduction residual is itself expanded into the actual tree's add
primitives. It is not a free final-output approximation assumption. -/
theorem transition_returned_error (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (q : Nat) (hq : q < route.components * d.dim) (ideal : Nat → ℝ) :
    let views := backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
      report.path.contiguousGradient report.path.gs report.path.gx report.path.gl
    let values := BatchContributions.value codec route graph ops d views report.dispatch.start.heap q
    real (load codec after.heap (returned.2.2.view.address q)) - (Finset.range d.batch).sum ideal =
      (Finset.range d.batch).sum (fun b => real (values b) - ideal b) +
        (sumExpr (report.sum.plan.tree q) values).error ops := by
  dsimp only
  rw [report.transitionReadback q hq, ← sumExpr_evaluate, Expr.error_exact]
  rw [sumExpr_plan report.sum.batch report.sum.dim report.path.gl.route.components report.sum.plan q
    (by simpa [report.sumRoute, report.sumDim] using hq)]
  rw [congrArg Finset.range report.sumBatch, Finset.sum_sub_distrib]
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.OutputNumerics
