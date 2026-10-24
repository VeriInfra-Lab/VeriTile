import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Reduction
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceActions

/-! The batch reduction's leaves are the actual grad_lambda temporary written
by the backward source grid. No reduction-result law is assumed here. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.BatchContributions
open Kernels ByteMemory Execution
set_option autoImplicit false

def column (route : Route) (d : Dimensions) (q b : Nat) := b * d.dim + q / route.components

def lane (c : Nat) : Lane := ⟨c % 128, Nat.mod_lt _ (by decide)⟩

def value (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (m : Heap) (q b : Nat) : Word :=
  let c := column route d q b
  getComponent (terminalGradient route ops
    (backwardValues codec route graph ops d views (c / 128) m d.length (lane c)).transition)
    (q % route.components)

theorem component_complete (route : Route) (k : Nat) (hk : k < route.components) : k ∈ componentList route := by
  cases route <;> simp [Route.components, componentList] at * <;> omega

theorem coordinate (route : Route) (d : Dimensions) (q b : Nat)
    (hq : q < route.components * d.dim) (hb : b < d.batch) :
    column route d q b < d.width ∧
      Reduction.inputWord route d.dim q b = route.components * column route d q b + q % route.components := by
  have hc : 0 < route.components := by cases route <;> decide
  have qd : q / route.components < d.dim := by
    apply (Nat.div_lt_iff_lt_mul hc).2
    simpa [Nat.mul_comm] using hq
  have mod := Nat.div_add_mod q route.components
  constructor
  · dsimp [column, Dimensions.width]
    nlinarith
  · dsimp [column, Reduction.inputWord]
    nlinarith

theorem grid_readback (codec : Codec Word) (route : Route) (graph : Graph) (ops : Operations Word)
    (d : Dimensions) (views : BackwardViews) (schedule : SourceActions.Schedule d) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardAllFresh views)
    (q b : Nat) (hq : q < route.components * d.dim) (hb : b < d.batch) :
    load codec (backwardGrid codec route graph ops d views schedule.order m)
      (views.gradTransition.address (Reduction.inputWord route d.dim q b)) =
        value codec route graph ops d views m q b := by
  have hc : 0 < route.components := by cases route <;> decide
  have coords := coordinate route d q b hq hb
  have owner : columnIndex (column route d q b / 128) (lane (column route d q b)) = column route d q b :=
    NativeIndex.column_owner _
  have activeLane : active d.width (column route d q b / 128) (lane (column route d q b)) := by
    change columnIndex _ _ < _
    rw [owner]
    exact coords.1
  rw [coords.2, ← owner]
  exact backwardGrid_transition codec route graph ops d views schedule.order _ m sizes fresh
    (SourceActions.owner_in_schedule d schedule _ _ activeLane) _ activeLane _
    (component_complete route _ (Nat.mod_lt q hc))

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.BatchContributions
