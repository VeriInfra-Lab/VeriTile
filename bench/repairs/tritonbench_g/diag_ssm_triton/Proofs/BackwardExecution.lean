import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ForwardExecution

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 4000000

def previousView (d : Dimensions) (views : BackwardViews) (i : Nat) : View :=
  if d.length - 1 - i = 0 then views.initial else views.output

def previousIndex (d : Dimensions) (pid i : Nat) : Lane → Nat :=
  if d.length - 1 - i = 0 then columnIndex pid else timeIndex d (d.length - 1 - i - 1) pid

def backwardUpdate {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid i : Nat) (transition : Tile α)
    (registers : Lane → BackwardRegisters α) (m : Heap) : Lane → BackwardStep α :=
  fun j => Kernels.backward route graph ops
    (readTile codec route ops (previousView d views i) d.width pid (previousIndex d pid i) m j)
    (transition j)
    (readTile codec route ops views.gradOutput d.width pid (timeIndex d (d.length - 1 - i) pid) m j)
    (registers j)

/-- The backward trajectory reads the actual saved output bytes. It is not
defined by replacing those bytes with the ideal forward trajectory. -/
def backwardValues {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) : Nat → Lane → BackwardRegisters α
  | 0 => fun _ => initialBackward ops
  | i + 1 => fun j => (backwardUpdate codec route graph ops d views pid i
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
      (backwardValues codec route graph ops d views pid m i) m j).registers

def backwardInputValues {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) (i : Nat) : Tile α :=
  fun j => (backwardUpdate codec route graph ops d views pid i
    (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
    (backwardValues codec route graph ops d views pid m i) m j).inputGradient

def backwardPrefix {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid k : Nat) (m : Heap) : ReverseMachine α :=
  backwardLoop codec route graph ops d views pid
    (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m) k
    (backwardInitial route ops d views pid m)

theorem previous_fits (route : Route) (d : Dimensions) (views : BackwardViews) (pid i : Nat)
    (initialSize : views.initial.words = route.components * d.width)
    (outputSize : views.output.words = route.components * (d.length * d.width)) (hi : i < d.length) :
    TileFits route (previousView d views i) d.width pid (previousIndex d pid i) := by
  unfold previousView previousIndex
  split
  · exact static_fits route views.initial d.width pid initialSize
  · exact time_fits route views.output d _ pid outputSize (by omega)

theorem backwardPrefix_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid k : Nat) (m : Heap) (caller : Backing)
    (fits : ∀ i < k, TileFits route views.gradInput d.width pid (timeIndex d (d.length - 1 - i) pid))
    (fresh : views.gradInput.backing.disjoint caller) :
    Frame m (backwardPrefix codec route graph ops d views pid k m).heap caller :=
  backwardLoop_frame codec route graph ops d views pid _ k _ caller fits fresh

structure BackwardFits (route : Route) (d : Dimensions) (views : BackwardViews) (pid steps : Nat) : Prop where
  gradInput : ∀ i < steps, TileFits route views.gradInput d.width pid (timeIndex d (d.length - 1 - i) pid)
  gradOutput : ∀ i < steps, TileFits route views.gradOutput d.width pid (timeIndex d (d.length - 1 - i) pid)
  previous : ∀ i < steps, TileFits route (previousView d views i) d.width pid (previousIndex d pid i)

theorem BackwardFits.prefix {route : Route} {d : Dimensions} {views : BackwardViews} {pid steps k : Nat}
    (h : BackwardFits route d views pid steps) (hk : k ≤ steps) : BackwardFits route d views pid k :=
  ⟨fun i hi => h.gradInput i (by omega), fun i hi => h.gradOutput i (by omega),
    fun i hi => h.previous i (by omega)⟩

structure BackwardFresh (views : BackwardViews) : Prop where
  gradOutput : views.gradInput.backing.disjoint views.gradOutput.backing
  initial : views.gradInput.backing.disjoint views.initial.backing
  output : views.gradInput.backing.disjoint views.output.backing

theorem BackwardFresh.previous {views : BackwardViews} (h : BackwardFresh views) (d : Dimensions) (i : Nat) :
    views.gradInput.backing.disjoint (previousView d views i).backing := by
  unfold previousView
  split
  · exact h.initial
  · exact h.output

theorem backwardUpdate_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid i : Nat) (transition : Tile α)
    (registers : Lane → BackwardRegisters α) (before after : Heap)
    (gyFits : TileFits route views.gradOutput d.width pid (timeIndex d (d.length - 1 - i) pid))
    (previousFits : TileFits route (previousView d views i) d.width pid (previousIndex d pid i))
    (gyFrame : Frame before after views.gradOutput.backing)
    (previousFrame : Frame before after (previousView d views i).backing) :
    backwardUpdate codec route graph ops d views pid i transition registers after =
      backwardUpdate codec route graph ops d views pid i transition registers before := by
  unfold backwardUpdate
  rw [readTile_frame codec route ops _ _ _ _ before after previousFits previousFrame,
    readTile_frame codec route ops _ _ _ _ before after gyFits gyFrame]

theorem backwardPrefix_values {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid k : Nat) (m : Heap)
    (fits : BackwardFits route d views pid k) (fresh : BackwardFresh views) :
    (backwardPrefix codec route graph ops d views pid k m).registers =
      backwardValues codec route graph ops d views pid m k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have ih := ih (fits.prefix (by omega))
    have gyFrame := backwardPrefix_frame codec route graph ops d views pid k m views.gradOutput.backing
      (fits.prefix (by omega)).gradInput fresh.gradOutput
    have previousFrame := backwardPrefix_frame codec route graph ops d views pid k m (previousView d views k).backing
      (fits.prefix (by omega)).gradInput (fresh.previous d k)
    have reads := backwardUpdate_frame codec route graph ops d views pid k
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
      (backwardValues codec route graph ops d views pid m k) m
      (backwardPrefix codec route graph ops d views pid k m).heap
      (fits.gradOutput k (by omega)) (fits.previous k (by omega)) gyFrame previousFrame
    change (fun j => (backwardUpdate codec route graph ops d views pid k
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
      (backwardPrefix codec route graph ops d views pid k m).registers
      (backwardPrefix codec route graph ops d views pid k m).heap j).registers) = _
    rw [ih, reads]
    rfl

theorem backwardPrefix_heap_step {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid k : Nat) (m : Heap)
    (fits : BackwardFits route d views pid (k + 1)) (fresh : BackwardFresh views) :
    (backwardPrefix codec route graph ops d views pid (k + 1) m).heap =
      storeTile codec route views.gradInput d.width pid (timeIndex d (d.length - 1 - k) pid)
        (backwardInputValues codec route graph ops d views pid m k)
        (backwardPrefix codec route graph ops d views pid k m).heap := by
  have registers := backwardPrefix_values codec route graph ops d views pid k m (fits.prefix (by omega)) fresh
  have gyFrame := backwardPrefix_frame codec route graph ops d views pid k m views.gradOutput.backing
    (fits.prefix (by omega)).gradInput fresh.gradOutput
  have previousFrame := backwardPrefix_frame codec route graph ops d views pid k m (previousView d views k).backing
    (fits.prefix (by omega)).gradInput (fresh.previous d k)
  have reads := backwardUpdate_frame codec route graph ops d views pid k
    (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
    (backwardValues codec route graph ops d views pid m k) m
    (backwardPrefix codec route graph ops d views pid k m).heap
    (fits.gradOutput k (by omega)) (fits.previous k (by omega)) gyFrame previousFrame
  change storeTile codec route views.gradInput d.width pid (timeIndex d (d.length - 1 - k) pid)
    (fun j => (backwardUpdate codec route graph ops d views pid k
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m)
      (backwardPrefix codec route graph ops d views pid k m).registers
      (backwardPrefix codec route graph ops d views pid k m).heap j).inputGradient)
    (backwardPrefix codec route graph ops d views pid k m).heap = _
  rw [registers, reads]
  rfl

theorem backwardPrefix_output {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid steps : Nat) (m : Heap)
    (fits : BackwardFits route d views pid steps) (fresh : BackwardFresh views)
    (stepsBound : steps ≤ d.length) (i : Nat) (hi : i < steps)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardPrefix codec route graph ops d views pid steps m).heap
      (views.gradInput.address (route.components * timeIndex d (d.length - 1 - i) pid j + k)) =
      getComponent (backwardInputValues codec route graph ops d views pid m i j) k := by
  induction steps with
  | zero => omega
  | succ steps ih =>
    rw [backwardPrefix_heap_step codec route graph ops d views pid steps m fits fresh]
    by_cases same : i = steps
    · subst i
      exact storeTile_read codec route views.gradInput d.width pid
        (timeIndex d (d.length - 1 - steps) pid) _ _ j hj k hk
        (by intro a b _ _ he; exact timeIndex_injective d _ pid a b he)
    · rw [storeTile_previous_time codec route views.gradInput d pid
        (d.length - 1 - steps) (d.length - 1 - i) _ _ j hj k hk (by omega)]
      exact ih (fits.prefix (by omega)) (by omega) (by omega)

/-- Terminal writes are present even when length=0. The transition-gradient
view contains the batch contributions before the later Torch dim0 reduction. -/
theorem backwardTerminal_transition {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (current : ReverseMachine α)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardTerminal codec route ops d views pid current).heap
      (views.gradTransition.address (route.components * columnIndex pid j + k)) =
      getComponent (terminalGradient route ops (current.registers j).transition) k := by
  apply storeTile_read codec route views.gradTransition d.width pid (columnIndex pid) _ _ j hj k hk
  intro a b _ _ same
  apply Fin.ext
  simp only [columnIndex, NativeIndex.column] at same
  omega

theorem backwardTerminal_initial {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (current : ReverseMachine α)
    (sFits : TileFits route views.gradInitial d.width pid (columnIndex pid))
    (lFits : TileFits route views.gradTransition d.width pid (columnIndex pid))
    (fresh : views.gradTransition.backing.disjoint views.gradInitial.backing)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardTerminal codec route ops d views pid current).heap
      (views.gradInitial.address (route.components * columnIndex pid j + k)) =
      getComponent (terminalGradient route ops (current.registers j).carry) k := by
  have frame := storeTile_frame codec route views.gradTransition d.width pid (columnIndex pid)
    (fun j => terminalGradient route ops (current.registers j).transition)
    (storeTile codec route views.gradInitial d.width pid (columnIndex pid)
      (fun j => terminalGradient route ops (current.registers j).carry) current.heap)
    views.gradInitial.backing lFits fresh
  have hi : route.components * columnIndex pid j + k < views.gradInitial.words := by
    have hf := sFits j hj
    have hc := component_member route k hk
    omega
  dsimp only [backwardTerminal]
  rw [Frame.load codec views.gradInitial frame _ hi]
  apply storeTile_read codec route views.gradInitial d.width pid (columnIndex pid) _ _ j hj k hk
  intro a b _ _ same
  apply Fin.ext
  simp only [columnIndex, NativeIndex.column] at same
  omega

theorem backwardTerminal_frame {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (current : ReverseMachine α) (caller : Backing)
    (sFits : TileFits route views.gradInitial d.width pid (columnIndex pid))
    (lFits : TileFits route views.gradTransition d.width pid (columnIndex pid))
    (freshS : views.gradInitial.backing.disjoint caller) (freshL : views.gradTransition.backing.disjoint caller) :
    Frame current.heap (backwardTerminal codec route ops d views pid current).heap caller := by
  exact (storeTile_frame codec route views.gradInitial d.width pid (columnIndex pid) _ _ caller sFits freshS).trans
    (storeTile_frame codec route views.gradTransition d.width pid (columnIndex pid) _ _ caller lFits freshL)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
