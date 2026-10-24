import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.OperandDomains

namespace VeriTile.Bench.Repairs.DiagSsmTriton.FiniteOutputs
open Kernels ByteMemory Execution Protocol Tensors Host LibraryContracts HostPaths Dispatch
open PrimitiveNumerics GraphNumerics TrajectoryNumerics OutputNumerics OperandDomains
set_option autoImplicit false
noncomputable section

theorem component (ops : Operations Word) (route : Route) (expression : Pair Expr)
    (domain : PairDomain ops route expression) (k : Nat) (hk : k ∈ componentList route) :
    Finite (getComponent (evalPair ops expression) k) := by
  cases route
  · have hk : k = 0 := by simpa [componentList] using hk
    subst k
    exact domain.1.2
  · have hk : k = 0 ∨ k = 1 := by simpa [componentList] using hk
    rcases hk with rfl | rfl
    · exact domain.1.2
    · exact (domain.2 rfl).2

theorem forward_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions)
    (s x transition returned : Tensor) (before after : State)
    (report : ForwardWrapper.Report codec graph ops api choices route d s x transition returned before after)
    (numerical : OperandDomains.forward codec route graph ops d
      (forwardViews report.path.initial report.path.input report.path.parameter returned) report.dispatch.start.heap)
    (t : Nat) (ht : t < d.length) (pid : Nat) (j : Lane) (hj : active d.width pid j)
    (k : Nat) (hk : k ∈ componentList route) :
    Finite (load codec after.heap (returned.view.address (route.components * timeIndex d t pid j + k))) := by
  have domain := numerical pid (NativeIndex.column_of_active d.width pid j.val j.isLt hj) j t ht
  have value := component ops route _ domain k hk
  rw [report.readback t ht pid j hj k hk]
  simpa only [eval_forward, forwardValues] using value

theorem input_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (numerical : OperandDomains.backward codec route graph ops d
      (backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
        report.path.contiguousGradient report.path.gs report.path.gx report.path.gl) report.dispatch.start.heap)
    (i : Nat) (hi : i < d.length) (pid : Nat) (j : Lane) (hj : active d.width pid j)
    (k : Nat) (hk : k ∈ componentList route) :
    Finite (load codec after.heap (returned.2.1.view.address (route.components * timeIndex d (d.length - 1 - i) pid j + k))) := by
  have domain := (numerical pid (NativeIndex.column_of_active d.width pid j.val j.isLt hj) j).1 i hi
  have value := component ops route _ domain.1 k hk
  rw [report.inputReadback i hi pid j hj k hk]
  unfold backwardExpression at value
  rw [(eval_backward ops route graph _ _ _ _).1] at value
  exact value

theorem initial_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (numerical : OperandDomains.backward codec route graph ops d
      (backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
        report.path.contiguousGradient report.path.gs report.path.gx report.path.gl) report.dispatch.start.heap)
    (pid : Nat) (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    Finite (load codec after.heap (returned.1.view.address (route.components * columnIndex pid j + k))) := by
  have domain := (numerical pid (NativeIndex.column_of_active d.width pid j.val j.isLt hj) j).2.1
  have value := component ops route _ domain k hk
  rw [report.initialReadback pid j hj k hk]
  have eval (pair : Pair Word) : evalPair ops (terminalExpr route pair) = terminalGradient route ops pair := by
    cases route <;> rfl
  rw [eval] at value
  exact value

theorem transition_word (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (api : Services) (choices : Choices) (route : Route) (d : Dimensions) (saved : Saved)
    (gy : Tensor) (returned : Tensor × Tensor × Tensor) (before after : State)
    (report : BackwardWrapper.Report codec graph ops api choices route d saved gy returned before after)
    (numerical : OperandDomains.reduction ops report.sum.batch report.sum.dim report.path.gl.route.components report.sum.plan
      (BatchContributions.value codec route graph ops d
        (backwardViews report.path.restored.1 report.path.restored.2.1 report.path.restored.2.2
          report.path.contiguousGradient report.path.gs report.path.gx report.path.gl) report.dispatch.start.heap))
    (q : Nat) (hq : q < route.components * d.dim) :
    Finite (load codec after.heap (returned.2.2.view.address q)) := by
  have domain := numerical q (by simpa [report.sumRoute, report.sumDim] using hq)
  rw [report.transitionReadback q hq]
  simpa only [sumExpr_evaluate] using domain.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.FiniteOutputs
