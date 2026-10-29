import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Failure

/-! A conservative failure envelope for concurrently progressing programs and
warps. Each retained byte store comes from an actual source-computed tile;
arbitrary interleavings and subsets of those stores are permitted. The runtime
chooses only event locations and ordering, never values or frame laws. Thus a
failure need not have only one partially completed program. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.InterruptedGrid
open Kernels ByteMemory Execution
set_option autoImplicit false

structure ForwardEvent (d : Dimensions) where
  pid : Nat
  valid : pid < NativeIndex.grid d.width
  time : Nat
  bound : time < d.length
  stores : List Nat

structure BackwardEvent (d : Dimensions) where
  pid : Nat
  valid : pid < NativeIndex.grid d.width
  point : Failure.ReversePoint d

def forwardEvent {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (event : ForwardEvent d) (initial current : Heap) : Heap :=
  Failure.partialTile codec route views.output d.width event.pid (timeIndex d event.time event.pid)
    (forwardPrefix codec route graph ops d views event.pid (event.time + 1) initial).registers event.stores current

def backwardEvent {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (event : BackwardEvent d) (initial current : Heap) : Heap :=
  match event.point with
  | .time i selection _ =>
    let previous := backwardPrefix codec route graph ops d views event.pid i initial
    let transition := readTile codec route ops views.transition d.width event.pid (transitionIndex d event.pid) initial
    let next := backwardUpdate codec route graph ops d views event.pid i transition previous.registers previous.heap
    Failure.partialTile codec route views.gradInput d.width event.pid
      (timeIndex d (d.length - 1 - i) event.pid) (fun j => (next j).inputGradient) selection current
  | .initial selection =>
    Failure.partialTile codec route views.gradInitial d.width event.pid (columnIndex event.pid)
      (fun j => terminalGradient route ops
        ((backwardPrefix codec route graph ops d views event.pid d.length initial).registers j).carry) selection current
  | .transition selection =>
    Failure.partialTile codec route views.gradTransition d.width event.pid (columnIndex event.pid)
      (fun j => terminalGradient route ops
        ((backwardPrefix codec route graph ops d views event.pid d.length initial).registers j).transition) selection current

def forward {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (events : List (ForwardEvent d)) (initial : Heap) : Heap :=
  events.foldl (fun current event => forwardEvent codec route graph ops d views event initial current) initial

def backward {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (events : List (BackwardEvent d)) (initial : Heap) : Heap :=
  events.foldl (fun current event => backwardEvent codec route graph ops d views event initial current) initial

theorem forward_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (events : List (ForwardEvent d)) (initial : Heap)
    (sizes : ForwardSizes route d views) (caller : Backing) (fresh : views.output.backing.disjoint caller) :
    Frame initial (forward codec route graph ops d views events initial) caller := by
  have step (event : ForwardEvent d) (current : Heap) :
      Frame current (forwardEvent codec route graph ops d views event initial current) caller :=
    Failure.partialTile_frame codec route views.output d.width event.pid (timeIndex d event.time event.pid)
      _ event.stores current caller (time_fits route views.output d event.time event.pid sizes.output event.bound) fresh
  have fold (es : List (ForwardEvent d)) (current : Heap) :
      Frame current (es.foldl (fun m e => forwardEvent codec route graph ops d views e initial m) current) caller := by
    induction es generalizing current with
    | nil => exact Frame.refl _ _
    | cons event rest ih => exact (step event current).trans (ih _)
  exact fold events initial

theorem backward_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (events : List (BackwardEvent d)) (initial : Heap)
    (sizes : BackwardSizes route d views) (caller : Backing)
    (freshX : views.gradInput.backing.disjoint caller) (freshS : views.gradInitial.backing.disjoint caller)
    (freshL : views.gradTransition.backing.disjoint caller) :
    Frame initial (backward codec route graph ops d views events initial) caller := by
  have step (event : BackwardEvent d) (current : Heap) :
      Frame current (backwardEvent codec route graph ops d views event initial current) caller := by
    cases h : event.point with
    | time i selection bound =>
      simp only [backwardEvent, h]
      exact Failure.partialTile_frame codec route views.gradInput d.width event.pid
        (timeIndex d (d.length - 1 - i) event.pid) _ selection current caller
        (time_fits route views.gradInput d _ event.pid sizes.gradInput (by omega)) freshX
    | initial selection =>
      simp only [backwardEvent, h]
      exact Failure.partialTile_frame codec route views.gradInitial d.width event.pid (columnIndex event.pid)
        _ selection current caller (static_fits route views.gradInitial d.width event.pid sizes.gradInitial) freshS
    | transition selection =>
      simp only [backwardEvent, h]
      exact Failure.partialTile_frame codec route views.gradTransition d.width event.pid (columnIndex event.pid)
        _ selection current caller (static_fits route views.gradTransition d.width event.pid sizes.gradTransition) freshL
  have fold (es : List (BackwardEvent d)) (current : Heap) :
      Frame current (es.foldl (fun m e => backwardEvent codec route graph ops d views e initial m) current) caller := by
    induction es generalizing current with
    | nil => exact Frame.refl _ _
    | cons event rest ih => exact (step event current).trans (ih _)
  exact fold events initial

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.InterruptedGrid
