import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardExecution

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 4000000

structure BackwardSizes (route : Route) (d : Dimensions) (views : BackwardViews) : Prop where
  initial : views.initial.words = route.components * d.width
  transition : views.transition.words = route.components * d.dim
  output : views.output.words = route.components * (d.length * d.width)
  gradInitial : views.gradInitial.words = route.components * d.width
  gradInput : views.gradInput.words = route.components * (d.length * d.width)
  gradTransition : views.gradTransition.words = route.components * d.width
  gradOutput : views.gradOutput.words = route.components * (d.length * d.width)

theorem BackwardSizes.fits {route : Route} {d : Dimensions} {views : BackwardViews}
    (h : BackwardSizes route d views) (pid : Nat) : BackwardFits route d views pid d.length := by
  constructor
  · intro i hi
    exact time_fits route views.gradInput d _ pid h.gradInput (by omega)
  · intro i hi
    exact time_fits route views.gradOutput d _ pid h.gradOutput (by omega)
  · intro i hi
    exact previous_fits route d views pid i h.initial h.output hi

structure OutputSeparation (views : BackwardViews) : Prop where
  initial_input : views.gradInitial.backing.disjoint views.gradInput.backing
  transition_input : views.gradTransition.backing.disjoint views.gradInput.backing
  transition_initial : views.gradTransition.backing.disjoint views.gradInitial.backing

theorem backwardProgram_input {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardFresh views) (outputs : OutputSeparation views)
    (i : Nat) (hi : i < d.length) (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradInput.address (route.components * timeIndex d (d.length - 1 - i) pid j + k)) =
      getComponent (backwardInputValues codec route graph ops d views pid m i j) k := by
  have frame := backwardTerminal_frame codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m) views.gradInput.backing
    (static_fits route views.gradInitial d.width pid sizes.gradInitial)
    (static_fits route views.gradTransition d.width pid sizes.gradTransition)
    outputs.initial_input outputs.transition_input
  have bound : route.components * timeIndex d (d.length - 1 - i) pid j + k < views.gradInput.words := by
    have fits := (sizes.fits pid).gradInput i hi j hj
    have hc := component_member route k hk
    omega
  change load codec (backwardTerminal codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m)).heap _ = _
  rw [Frame.load codec views.gradInput frame _ bound]
  exact backwardPrefix_output codec route graph ops d views pid d.length m (sizes.fits pid) fresh le_rfl i hi j hj k hk

theorem backwardProgram_initial {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardFresh views) (outputs : OutputSeparation views)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradInitial.address (route.components * columnIndex pid j + k)) =
      getComponent (terminalGradient route ops (backwardValues codec route graph ops d views pid m d.length j).carry) k := by
  change load codec (backwardTerminal codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m)).heap _ = _
  rw [backwardTerminal_initial codec route ops d views pid _
    (static_fits route views.gradInitial d.width pid sizes.gradInitial)
    (static_fits route views.gradTransition d.width pid sizes.gradTransition)
    outputs.transition_initial j hj k hk]
  rw [backwardPrefix_values codec route graph ops d views pid d.length m (sizes.fits pid) fresh]

theorem backwardProgram_transition {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardFresh views)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradTransition.address (route.components * columnIndex pid j + k)) =
      getComponent (terminalGradient route ops (backwardValues codec route graph ops d views pid m d.length j).transition) k := by
  change load codec (backwardTerminal codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m)).heap _ = _
  rw [backwardTerminal_transition codec route ops d views pid _ j hj k hk,
    backwardPrefix_values codec route graph ops d views pid d.length m (sizes.fits pid) fresh]

theorem backwardProgram_caller_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (caller : Backing)
    (freshX : views.gradInput.backing.disjoint caller)
    (freshS : views.gradInitial.backing.disjoint caller)
    (freshL : views.gradTransition.backing.disjoint caller) :
    Frame m (backwardProgram codec route graph ops d views pid m).heap caller := by
  apply (backwardPrefix_frame codec route graph ops d views pid d.length m caller
    (sizes.fits pid).gradInput freshX).trans
  exact backwardTerminal_frame codec route ops d views pid _ caller
    (static_fits route views.gradInitial d.width pid sizes.gradInitial)
    (static_fits route views.gradTransition d.width pid sizes.gradTransition) freshS freshL

theorem backwardPrefix_full_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid steps : Nat) (m : Heap)
    (fits : ∀ i < steps, TileFits route views.gradInput d.width pid (timeIndex d (d.length - 1 - i) pid))
    (p : Nat) (outside : p < views.gradInput.pointer ∨ views.gradInput.pointer + 4 * views.gradInput.words ≤ p) :
    (backwardPrefix codec route graph ops d views pid steps m).heap p = m p := by
  induction steps with
  | zero => rfl
  | succ steps ih =>
    change storeTile codec route views.gradInput d.width pid (timeIndex d (d.length - 1 - steps) pid) _
      (backwardPrefix codec route graph ops d views pid steps m).heap p = _
    rw [storeTile_outside_view codec route views.gradInput d.width pid
      (timeIndex d (d.length - 1 - steps) pid) _ _ (fits steps (by omega)) p outside]
    exact ih (by intro i hi; exact fits i (by omega))

def OutsideBackward (views : BackwardViews) (p : Nat) : Prop :=
  (p < views.gradInput.pointer ∨ views.gradInput.pointer + 4 * views.gradInput.words ≤ p) ∧
  (p < views.gradInitial.pointer ∨ views.gradInitial.pointer + 4 * views.gradInitial.words ≤ p) ∧
  (p < views.gradTransition.pointer ∨ views.gradTransition.pointer + 4 * views.gradTransition.words ≤ p)

theorem backwardProgram_full_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (p : Nat) (outside : OutsideBackward views p) :
    (backwardProgram codec route graph ops d views pid m).heap p = m p := by
  change storeTile codec route views.gradTransition d.width pid (columnIndex pid) _
    (storeTile codec route views.gradInitial d.width pid (columnIndex pid) _
      (backwardPrefix codec route graph ops d views pid d.length m).heap) p = _
  rw [storeTile_outside_view codec route views.gradTransition d.width pid (columnIndex pid) _ _
      (static_fits route views.gradTransition d.width pid sizes.gradTransition) p outside.2.2,
    storeTile_outside_view codec route views.gradInitial d.width pid (columnIndex pid) _ _
      (static_fits route views.gradInitial d.width pid sizes.gradInitial) p outside.2.1]
  exact backwardPrefix_full_frame codec route graph ops d views pid d.length m (sizes.fits pid).gradInput p outside.1

theorem backwardGrid_full_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (order : List Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (p : Nat) (outside : OutsideBackward views p) :
    backwardGrid codec route graph ops d views order m p = m p := by
  induction order generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    change backwardGrid codec route graph ops d views rest (backwardProgram codec route graph ops d views pid m).heap p = _
    rw [ih]
    exact backwardProgram_full_frame codec route graph ops d views pid m sizes p outside

theorem backwardGrid_caller_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (order : List Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (caller : Backing)
    (freshX : views.gradInput.backing.disjoint caller)
    (freshS : views.gradInitial.backing.disjoint caller)
    (freshL : views.gradTransition.backing.disjoint caller) :
    Frame m (backwardGrid codec route graph ops d views order m) caller := by
  induction order generalizing m with
  | nil => exact Frame.refl _ _
  | cons pid rest ih =>
    exact (backwardProgram_caller_frame codec route graph ops d views pid m sizes caller freshX freshS freshL).trans (ih _)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
