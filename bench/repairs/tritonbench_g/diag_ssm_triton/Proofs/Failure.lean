import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardComplete
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ForwardGrid

/-! Actual partial kernel writes. A runtime failure may follow completed
programs, completed time steps, and an arbitrary selection/order of the current
masked component stores. This includes partially completed vector instructions.
The earlier writes remain in the heap. Frames are derived from those
stores; no failed-launch memory oracle or rollback rule is assumed. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Failure
open Kernels ByteMemory Execution

def partialTile {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (selection : List Nat) (m : Heap) : Heap :=
  stores codec (selection.filterMap (fun i => (tileStores route view n pid index values)[i]?)) m

theorem partialTile_frame {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (selection : List Nat) (m : Heap)
    (caller : Backing) (fits : TileFits route view n pid index) (fresh : view.backing.disjoint caller) :
    Frame m (partialTile codec route view n pid index values selection m) caller := by
  intro p hp
  apply stores_frame
  intro event he
  obtain ⟨i, _, lookup⟩ := List.mem_filterMap.mp he
  have he := List.mem_of_getElem? lookup
  obtain ⟨k, hk, he⟩ := List.mem_flatMap.mp he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
  have hk := component_member route k hk
  have hf := fits j ((selected_member n pid j).1 hj)
  have hc := view.capacity
  simp only [Backing.contains] at hp
  simp only [Backing.disjoint] at fresh
  simp only [View.address, View.pointer]
  omega

structure ForwardPoint (d : Dimensions) where
  completed : List Nat
  pid : Nat
  steps : Nat
  stepBound : steps < d.length
  stores : List Nat

def forward {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (point : ForwardPoint d) (m : Heap) : Heap :=
  let grid := forwardGrid codec route graph ops d views point.completed m
  let beforeStep := forwardPrefix codec route graph ops d views point.pid point.steps grid
  let next := forwardPrefix codec route graph ops d views point.pid (point.steps + 1) grid
  partialTile codec route views.output d.width point.pid (timeIndex d point.steps point.pid)
    next.registers point.stores beforeStep.heap

theorem forward_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (point : ForwardPoint d) (m : Heap)
    (sizes : ForwardSizes route d views) (caller : Backing) (fresh : views.output.backing.disjoint caller) :
    Frame m (forward codec route graph ops d views point m) caller := by
  have grid := forwardGrid_caller_frame codec route graph ops d views point.completed m sizes caller fresh
  have first := forwardPrefix_frame codec route graph ops d views point.pid point.steps
    (forwardGrid codec route graph ops d views point.completed m) caller
    (fun t ht => time_fits route views.output d t point.pid sizes.output (by have := point.stepBound; omega)) fresh
  exact (grid.trans first).trans (partialTile_frame codec route views.output d.width point.pid
    (timeIndex d point.steps point.pid) _ point.stores _ caller
    (time_fits route views.output d point.steps point.pid sizes.output point.stepBound) fresh)

inductive ReversePoint (d : Dimensions) where
  | time (steps : Nat) (stores : List Nat) (stepBound : steps < d.length)
  | initial (stores : List Nat)
  | transition (stores : List Nat)

structure BackwardPoint (d : Dimensions) where
  completed : List Nat
  pid : Nat
  point : ReversePoint d

def backward {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (point : BackwardPoint d) (m : Heap) : Heap :=
  let grid := backwardGrid codec route graph ops d views point.completed m
  match point.point with
  | .time steps count _ =>
    let beforeStep := backwardPrefix codec route graph ops d views point.pid steps grid
    let transition := readTile codec route ops views.transition d.width point.pid (transitionIndex d point.pid) grid
    let next := backwardUpdate codec route graph ops d views point.pid steps transition beforeStep.registers beforeStep.heap
    partialTile codec route views.gradInput d.width point.pid (timeIndex d (d.length - 1 - steps) point.pid)
      (fun j => (next j).inputGradient) count beforeStep.heap
  | .initial count =>
    let beforeTerminal := backwardPrefix codec route graph ops d views point.pid d.length grid
    partialTile codec route views.gradInitial d.width point.pid (columnIndex point.pid)
      (fun j => terminalGradient route ops (beforeTerminal.registers j).carry) count beforeTerminal.heap
  | .transition count =>
    let beforeTerminal := backwardPrefix codec route graph ops d views point.pid d.length grid
    let afterS := storeTile codec route views.gradInitial d.width point.pid (columnIndex point.pid)
      (fun j => terminalGradient route ops (beforeTerminal.registers j).carry) beforeTerminal.heap
    partialTile codec route views.gradTransition d.width point.pid (columnIndex point.pid)
      (fun j => terminalGradient route ops (beforeTerminal.registers j).transition) count afterS

theorem backward_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (point : BackwardPoint d) (m : Heap)
    (sizes : BackwardSizes route d views) (caller : Backing)
    (freshX : views.gradInput.backing.disjoint caller)
    (freshS : views.gradInitial.backing.disjoint caller)
    (freshL : views.gradTransition.backing.disjoint caller) :
    Frame m (backward codec route graph ops d views point m) caller := by
  have grid := backwardGrid_caller_frame codec route graph ops d views point.completed m sizes caller freshX freshS freshL
  have before (steps : Nat) (hsteps : steps ≤ d.length) :=
    backwardPrefix_frame codec route graph ops d views point.pid steps
      (backwardGrid codec route graph ops d views point.completed m) caller
      ((sizes.fits point.pid).prefix hsteps).gradInput freshX
  cases hp : point.point with
  | time steps stores bound =>
    simp only [backward, hp]
    exact (grid.trans (before steps (by omega))).trans
      (partialTile_frame codec route views.gradInput d.width point.pid
        (timeIndex d (d.length - 1 - steps) point.pid) _ stores _ caller
        ((sizes.fits point.pid).gradInput steps bound) freshX)
  | initial stores =>
    simp only [backward, hp]
    exact (grid.trans (before d.length le_rfl)).trans
      (partialTile_frame codec route views.gradInitial d.width point.pid (columnIndex point.pid) _ stores _ caller
        (static_fits route views.gradInitial d.width point.pid sizes.gradInitial) freshS)
  | transition stores =>
    simp only [backward, hp]
    have first := storeTile_frame codec route views.gradInitial d.width point.pid (columnIndex point.pid)
      (fun j => terminalGradient route ops ((backwardPrefix codec route graph ops d views point.pid d.length
        (backwardGrid codec route graph ops d views point.completed m)).registers j).carry)
      (backwardPrefix codec route graph ops d views point.pid d.length
        (backwardGrid codec route graph ops d views point.completed m)).heap caller
      (static_fits route views.gradInitial d.width point.pid sizes.gradInitial) freshS
    exact ((grid.trans (before d.length le_rfl)).trans first).trans
      (partialTile_frame codec route views.gradTransition d.width point.pid (columnIndex point.pid) _ stores _ caller
        (static_fits route views.gradTransition d.width point.pid sizes.gradTransition) freshL)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Failure
