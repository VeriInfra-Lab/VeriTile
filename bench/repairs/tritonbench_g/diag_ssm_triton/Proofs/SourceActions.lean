import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Host
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.InterruptedGrid

/-! Concrete kernel actions for the host program. Runtime decisions choose a
dispatch order or actual source store events from an interrupted grid; they never supply output
values, a success equation, or a memory-frame law. Native grid-product and
segmentation correspondence is the separately deferred ordinary runtime edge.
All task-specific values and frames come from the checked source interpreters. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceActions
open Kernels ByteMemory Execution Protocol Host

structure Schedule (d : Dimensions) where
  order : List Nat
  unique : order.Nodup
  valid : ∀ pid ∈ order, pid < NativeIndex.grid d.width
  covers : ∀ pid, pid < NativeIndex.grid d.width → pid ∈ order

inductive ForwardDecision (d : Dimensions) where
  | complete (schedule : Schedule d)
  | failedSetup (error : Error)
  | stoppedDuringSource (point : List (InterruptedGrid.ForwardEvent d)) (error : Error)

inductive BackwardDecision (d : Dimensions) where
  | complete (schedule : Schedule d)
  | failedSetup (error : Error)
  | stoppedDuringSource (point : List (InterruptedGrid.BackwardEvent d)) (error : Error)

def forward (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choose : ∀ (route : Route) (d : Dimensions) (views : ForwardViews), State → ForwardDecision d)
    (route : Route) (d : Dimensions) (views : ForwardViews) : Program State Unit := fun state =>
  match choose route d views state with
  | .complete schedule => .success ()
      {state with heap := forwardGrid codec route graph ops d views schedule.order state.heap} []
  | .failedSetup error => .failure error state []
  | .stoppedDuringSource point error => .failure error
      {state with heap := InterruptedGrid.forward codec route graph ops d views point state.heap} []

def backward (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choose : ∀ (route : Route) (d : Dimensions) (views : BackwardViews), State → BackwardDecision d)
    (route : Route) (d : Dimensions) (views : BackwardViews) : Program State Unit := fun state =>
  match choose route d views state with
  | .complete schedule => .success ()
      {state with heap := backwardGrid codec route graph ops d views schedule.order state.heap} []
  | .failedSetup error => .failure error state []
  | .stoppedDuringSource point error => .failure error
      {state with heap := InterruptedGrid.backward codec route graph ops d views point state.heap} []

def kernels (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (forwardChoice : ∀ (route : Route) (d : Dimensions) (views : ForwardViews), State → ForwardDecision d)
    (backwardChoice : ∀ (route : Route) (d : Dimensions) (views : BackwardViews), State → BackwardDecision d) : KernelActions :=
  ⟨forward codec graph ops forwardChoice, backward codec graph ops backwardChoice⟩

theorem forward_preserves (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choose : ∀ (route : Route) (d : Dimensions) (views : ForwardViews), State → ForwardDecision d)
    (route : Route) (d : Dimensions) (views : ForwardViews)
    (sizes : ForwardSizes route d views) (caller : Backing) (fresh : views.output.backing.disjoint caller) :
    Preserves State.heap caller (forward codec graph ops choose route d views) := by
  intro state
  cases hc : choose route d views state with
  | complete schedule =>
    simp only [forward, hc, Result.state]
    exact forwardGrid_caller_frame codec route graph ops d views schedule.order state.heap sizes caller fresh
  | failedSetup error =>
    simp only [forward, hc, Result.state]
    exact Frame.refl _ _
  | stoppedDuringSource point error =>
    simp only [forward, hc, Result.state]
    exact InterruptedGrid.forward_frame codec route graph ops d views point state.heap sizes caller fresh

theorem backward_preserves (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choose : ∀ (route : Route) (d : Dimensions) (views : BackwardViews), State → BackwardDecision d)
    (route : Route) (d : Dimensions) (views : BackwardViews)
    (sizes : BackwardSizes route d views) (caller : Backing)
    (freshX : views.gradInput.backing.disjoint caller)
    (freshS : views.gradInitial.backing.disjoint caller)
    (freshL : views.gradTransition.backing.disjoint caller) :
    Preserves State.heap caller (backward codec graph ops choose route d views) := by
  intro state
  cases hc : choose route d views state with
  | complete schedule =>
    simp only [backward, hc, Result.state]
    exact backwardGrid_caller_frame codec route graph ops d views schedule.order state.heap sizes caller freshX freshS freshL
  | failedSetup error =>
    simp only [backward, hc, Result.state]
    exact Frame.refl _ _
  | stoppedDuringSource point error =>
    simp only [backward, hc, Result.state]
    exact InterruptedGrid.backward_frame codec route graph ops d views point state.heap sizes caller freshX freshS freshL

theorem owner_in_schedule (d : Dimensions) (schedule : Schedule d) (owner : Nat) (j : Lane)
    (activeLane : active d.width owner j) : owner ∈ schedule.order :=
  schedule.covers owner (NativeIndex.column_of_active d.width owner j.val j.isLt activeLane)

/-- No successful kernel equation is assumed: a complete runtime decision
executes the source and yields the actual heap and derived word readback. -/
theorem forward_complete (codec : Codec Word) (graph : Graph) (ops : Operations Word)
    (choose : ∀ (route : Route) (d : Dimensions) (views : ForwardViews), State → ForwardDecision d)
    (route : Route) (d : Dimensions) (views : ForwardViews) (state : State) (schedule : Schedule d)
    (decision : choose route d views state = .complete schedule)
    (sizes : ForwardSizes route d views) (fresh : ForwardFresh views)
    (t : Nat) (ht : t < d.length) (owner : Nat) (j : Lane) (hj : active d.width owner j)
    (k : Nat) (hk : k ∈ componentList route) :
    let after := {state with heap := forwardGrid codec route graph ops d views schedule.order state.heap}
    forward codec graph ops choose route d views state = .success () after [] ∧
      load codec after.heap (views.output.address (route.components * timeIndex d t owner j + k)) =
        getComponent (forwardValues codec route graph ops d views owner state.heap (t + 1) j) k := by
  constructor
  · simp [forward, decision]
  · exact forwardGrid_output codec route graph ops d views schedule.order owner t state.heap sizes fresh
      (owner_in_schedule d schedule owner j hj) ht j hj k hk

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceActions
