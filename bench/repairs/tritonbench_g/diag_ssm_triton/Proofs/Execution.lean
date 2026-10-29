import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ScalarGraphs
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ByteMemory
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.NativeIndex

/-! Source-ordered byte-memory execution. Every time iteration loads from the
current heap, computes the selected scalar graph, and performs the actual
masked component stores before the next iteration. Empty time preserves the
static loads and the backward terminal stores. No successful whole execution,
output value or memory-frame law is a backend premise. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 2000000

abbrev Lane := Fin 128
abbrev Tile (α : Type) := Lane → Pair α

def componentList : Route → List Nat
  | .real32 => [0]
  | .complex64 => [0, 1]

theorem component_member (route : Route) (k : Nat) (h : k ∈ componentList route) :
    k < route.components := by
  cases route <;> simp [componentList, Route.components] at * <;> omega

def getComponent {α : Type} (v : Pair α) (k : Nat) : α := if k = 0 then v.re else v.im

abbrev active (n pid : Nat) (j : Lane) := NativeIndex.column pid j.val < n
def selected (n pid : Nat) : List Lane := (List.finRange 128).filter (fun j => decide (active n pid j))

@[simp] theorem selected_member (n pid : Nat) (j : Lane) : j ∈ selected n pid ↔ active n pid j := by
  simp [selected]

inductive Access where
  | load (address : Nat)
  | store (address : Nat)
  deriving DecidableEq, Repr

def tileAccesses (kind : Nat → Access) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) : List Access :=
  (componentList route).flatMap (fun k =>
    (selected n pid).map (fun j => kind (view.address (route.components * index j + k))))

def readTile {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (view : View) (n pid : Nat) (index : Lane → Nat) (m : Heap) : Tile α :=
  fun j => if active n pid j then
    ⟨load codec m (view.address (route.components * index j)),
     match route with
     | .real32 => ops.zero
     | .complex64 => load codec m (view.address (route.components * index j + 1))⟩
  else zeroPair ops

def tileStores {α : Type} (route : Route) (view : View) (n pid : Nat)
    (index : Lane → Nat) (values : Tile α) : List (Nat × α) :=
  (componentList route).flatMap (fun k => (selected n pid).map
    (fun j => (view.address (route.components * index j + k), getComponent (values j) k)))

def storeTile {α : Type} (codec : Codec α) (route : Route) (view : View) (n pid : Nat)
    (index : Lane → Nat) (values : Tile α) (m : Heap) : Heap :=
  stores codec (tileStores route view n pid index values) m

def TileFits (route : Route) (view : View) (n pid : Nat) (index : Lane → Nat) : Prop :=
  ∀ j, active n pid j → route.components * index j + route.components ≤ view.words

theorem storeTile_frame {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (m : Heap) (caller : Backing)
    (fits : TileFits route view n pid index) (fresh : view.backing.disjoint caller) :
    Frame m (storeTile codec route view n pid index values m) caller := by
  intro p hp
  apply stores_frame
  intro event he
  obtain ⟨k, hk, he⟩ := List.mem_flatMap.mp he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
  have hj := (selected_member n pid j).1 hj
  have hk := component_member route k hk
  have hf := fits j hj
  have hi : route.components * index j + k < view.words := by omega
  have hc := view.capacity
  simp only [Backing.contains] at hp
  simp only [Backing.disjoint] at fresh
  simp only [View.address, View.pointer]
  omega

theorem readTile_frame {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (view : View) (n pid : Nat) (index : Lane → Nat) (before after : Heap)
    (fits : TileFits route view n pid index) (frame : Frame before after view.backing) :
    readTile codec route ops view n pid index after = readTile codec route ops view n pid index before := by
  funext j
  unfold readTile
  split
  · rename_i hj
    have hf := fits j hj
    cases route
    · congr 1
      exact Frame.load codec view frame _ (by simpa [Route.components] using hf)
    · congr 1
      · exact Frame.load codec view frame _ (by simp only [Route.components] at *; omega)
      · exact Frame.load codec view frame _ (by simp only [Route.components] at *; omega)
  · rfl

structure Dimensions where
  length : Nat
  batch : Nat
  dim : Nat

def Dimensions.width (d : Dimensions) := d.batch * d.dim
def columnIndex (pid : Nat) (j : Lane) := NativeIndex.column pid j.val
def timeIndex (d : Dimensions) (t pid : Nat) (j : Lane) := t * d.width + columnIndex pid j
def transitionIndex (d : Dimensions) (pid : Nat) (j : Lane) := columnIndex pid j % d.dim

structure ForwardViews where
  initial : View
  input : View
  transition : View
  output : View

structure BackwardViews where
  initial : View
  transition : View
  output : View
  gradInitial : View
  gradInput : View
  gradTransition : View
  gradOutput : View

structure Machine (α : Type) where
  heap : Heap
  trace : List Access
  registers : Tile α

def forwardInitial {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) : Machine α :=
  ⟨m,
   tileAccesses Access.load route views.initial d.width pid (columnIndex pid) ++
     tileAccesses Access.load route views.transition d.width pid (transitionIndex d pid),
   readTile codec route ops views.initial d.width pid (columnIndex pid) m⟩

def forwardStep {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid t : Nat) (transition : Tile α)
    (current : Machine α) : Machine α :=
  let index := timeIndex d t pid
  let input := readTile codec route ops views.input d.width pid index current.heap
  let next := fun j => Kernels.forward route graph ops (current.registers j) (transition j) (input j)
  ⟨storeTile codec route views.output d.width pid index next current.heap,
   current.trace ++ tileAccesses Access.load route views.input d.width pid index ++
     tileAccesses Access.store route views.output d.width pid index,
   next⟩

def forwardLoop {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (transition : Tile α) :
    Nat → Machine α → Machine α
  | 0, initial => initial
  | steps + 1, initial => forwardStep codec route graph ops d views pid steps transition
      (forwardLoop codec route graph ops d views pid transition steps initial)

def forwardProgram {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) : Machine α :=
  let transition := readTile codec route ops views.transition d.width pid (transitionIndex d pid) m
  forwardLoop codec route graph ops d views pid transition d.length
    (forwardInitial codec route ops d views pid m)

structure ReverseMachine (α : Type) where
  heap : Heap
  trace : List Access
  registers : Lane → BackwardRegisters α

def backwardInitial {α : Type} (route : Route) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) : ReverseMachine α :=
  ⟨m, tileAccesses Access.load route views.transition d.width pid (transitionIndex d pid),
    fun _ => initialBackward ops⟩

def backwardStep {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid i : Nat) (transition : Tile α)
    (current : ReverseMachine α) : ReverseMachine α :=
  let t := d.length - 1 - i
  let index := timeIndex d t pid
  let gy := readTile codec route ops views.gradOutput d.width pid index current.heap
  let previousView := if t = 0 then views.initial else views.output
  let previousIndex := if t = 0 then columnIndex pid else timeIndex d (t - 1) pid
  let previous := readTile codec route ops previousView d.width pid previousIndex current.heap
  let next := fun j => Kernels.backward route graph ops (previous j) (transition j) (gy j) (current.registers j)
  let gx := fun j => (next j).inputGradient
  ⟨storeTile codec route views.gradInput d.width pid index gx current.heap,
   current.trace ++ tileAccesses Access.load route views.gradOutput d.width pid index ++
     tileAccesses Access.load route previousView d.width pid previousIndex ++
     tileAccesses Access.store route views.gradInput d.width pid index,
   fun j => (next j).registers⟩

def backwardLoop {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (transition : Tile α) :
    Nat → ReverseMachine α → ReverseMachine α
  | 0, initial => initial
  | steps + 1, initial => backwardStep codec route graph ops d views pid steps transition
      (backwardLoop codec route graph ops d views pid transition steps initial)

def backwardTerminal {α : Type} (codec : Codec α) (route : Route) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (current : ReverseMachine α) : ReverseMachine α :=
  let gs := fun j => terminalGradient route ops (current.registers j).carry
  let gl := fun j => terminalGradient route ops (current.registers j).transition
  let afterS := storeTile codec route views.gradInitial d.width pid (columnIndex pid) gs current.heap
  let afterL := storeTile codec route views.gradTransition d.width pid (columnIndex pid) gl afterS
  ⟨afterL, current.trace ++ tileAccesses Access.store route views.gradInitial d.width pid (columnIndex pid) ++
      tileAccesses Access.store route views.gradTransition d.width pid (columnIndex pid), current.registers⟩

def backwardProgram {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap) : ReverseMachine α :=
  let transition := readTile codec route ops views.transition d.width pid (transitionIndex d pid) m
  backwardTerminal codec route ops d views pid
    (backwardLoop codec route graph ops d views pid transition d.length
      (backwardInitial route ops d views pid m))

/-- Every program consumes the preceding program's actual heap. -/
def forwardGrid {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (order : List Nat) (m : Heap) : Heap :=
  order.foldl (fun current pid => (forwardProgram codec route graph ops d views pid current).heap) m

def backwardGrid {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (order : List Nat) (m : Heap) : Heap :=
  order.foldl (fun current pid => (backwardProgram codec route graph ops d views pid current).heap) m

theorem forwardLoop_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (transition : Tile α)
    (steps : Nat) (initial : Machine α) (caller : Backing)
    (fits : ∀ t < steps, TileFits route views.output d.width pid (timeIndex d t pid))
    (fresh : views.output.backing.disjoint caller) :
    Frame initial.heap (forwardLoop codec route graph ops d views pid transition steps initial).heap caller := by
  induction steps with
  | zero => exact Frame.refl _ _
  | succ steps ih =>
    have prev := ih (by intro t ht; exact fits t (by omega))
    apply prev.trans
    exact storeTile_frame codec route views.output d.width pid (timeIndex d steps pid) _ _ caller
      (fits steps (by omega)) fresh

theorem backwardLoop_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (pid : Nat) (transition : Tile α)
    (steps : Nat) (initial : ReverseMachine α) (caller : Backing)
    (fits : ∀ i < steps, TileFits route views.gradInput d.width pid (timeIndex d (d.length - 1 - i) pid))
    (fresh : views.gradInput.backing.disjoint caller) :
    Frame initial.heap (backwardLoop codec route graph ops d views pid transition steps initial).heap caller := by
  induction steps with
  | zero => exact Frame.refl _ _
  | succ steps ih =>
    have prev := ih (by intro i hi; exact fits i (by omega))
    apply prev.trans
    exact storeTile_frame codec route views.gradInput d.width pid
      (timeIndex d (d.length - 1 - steps) pid) _ _ caller (fits steps (by omega)) fresh

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
