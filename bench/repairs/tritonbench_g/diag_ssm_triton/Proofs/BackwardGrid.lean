import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.BackwardComplete
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ForwardGrid

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 4000000

structure BackwardReadFrame (views : BackwardViews) (before after : Heap) : Prop where
  initial : Frame before after views.initial.backing
  transition : Frame before after views.transition.backing
  output : Frame before after views.output.backing
  gradOutput : Frame before after views.gradOutput.backing

def OutputsFreshFor (views : BackwardViews) (caller : Backing) : Prop :=
  views.gradInput.backing.disjoint caller ∧ views.gradInitial.backing.disjoint caller ∧
    views.gradTransition.backing.disjoint caller

structure BackwardAllFresh (views : BackwardViews) : Prop where
  initial : OutputsFreshFor views views.initial.backing
  transition : OutputsFreshFor views views.transition.backing
  output : OutputsFreshFor views views.output.backing
  gradOutput : OutputsFreshFor views views.gradOutput.backing
  outputs : OutputSeparation views

def BackwardAllFresh.prefix {views : BackwardViews} (h : BackwardAllFresh views) : BackwardFresh views :=
  ⟨h.gradOutput.1, h.initial.1, h.output.1⟩

theorem backwardProgram_read_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardAllFresh views) :
    BackwardReadFrame views m (backwardProgram codec route graph ops d views pid m).heap := by
  have frame (caller : Backing) (h : OutputsFreshFor views caller) :=
    backwardProgram_caller_frame codec route graph ops d views pid m sizes caller h.1 h.2.1 h.2.2
  exact ⟨frame _ fresh.initial, frame _ fresh.transition, frame _ fresh.output, frame _ fresh.gradOutput⟩

theorem BackwardReadFrame.previous {views : BackwardViews} {before after : Heap}
    (h : BackwardReadFrame views before after) (d : Dimensions) (i : Nat) :
    Frame before after (previousView d views i).backing := by
  unfold previousView
  split
  · exact h.initial
  · exact h.output

theorem backwardValues_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid steps : Nat) (before after : Heap)
    (sizes : BackwardSizes route d views) (frame : BackwardReadFrame views before after) (bound : steps ≤ d.length) :
    backwardValues codec route graph ops d views pid after steps =
      backwardValues codec route graph ops d views pid before steps := by
  induction steps with
  | zero => rfl
  | succ steps ih =>
    have ih := ih (by omega)
    have transition := readTile_frame codec route ops views.transition d.width pid (transitionIndex d pid) before after
      (transition_fits route views.transition d pid sizes.transition) frame.transition
    have read := backwardUpdate_frame codec route graph ops d views pid steps
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) before)
      (backwardValues codec route graph ops d views pid before steps) before after
      ((sizes.fits pid).gradOutput steps (by omega)) ((sizes.fits pid).previous steps (by omega))
      frame.gradOutput (frame.previous d steps)
    simp only [backwardValues, transition, ih, read]

theorem backwardInputValues_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid i : Nat) (before after : Heap)
    (sizes : BackwardSizes route d views) (frame : BackwardReadFrame views before after) (bound : i < d.length) :
    backwardInputValues codec route graph ops d views pid after i =
      backwardInputValues codec route graph ops d views pid before i := by
  have registers := backwardValues_frame codec route graph ops d views pid i before after sizes frame (by omega)
  have transition := readTile_frame codec route ops views.transition d.width pid (transitionIndex d pid) before after
    (transition_fits route views.transition d pid sizes.transition) frame.transition
  have read := backwardUpdate_frame codec route graph ops d views pid i
    (readTile codec route ops views.transition d.width pid (transitionIndex d pid) before)
    (backwardValues codec route graph ops d views pid before i) before after
    ((sizes.fits pid).gradOutput i bound) ((sizes.fits pid).previous i bound)
    frame.gradOutput (frame.previous d i)
  unfold backwardInputValues
  rw [transition, registers, read]

theorem backwardPrefix_other_program {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid steps owner t : Nat) (m : Heap)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) (different : owner ≠ pid) :
    load codec (backwardPrefix codec route graph ops d views pid steps m).heap
      (views.gradInput.address (route.components * timeIndex d t owner j + k)) =
      load codec m (views.gradInput.address (route.components * timeIndex d t owner j + k)) := by
  induction steps with
  | zero => rfl
  | succ steps ih =>
    change load codec (storeTile codec route views.gradInput d.width pid
      (timeIndex d (d.length - 1 - steps) pid) _
      (backwardPrefix codec route graph ops d views pid steps m).heap) _ = _
    rw [storeTile_load_outside]
    · exact ih
    · intro i hi l hl
      exact component_other_program route d owner pid t _ j i hj hi k l hk hl different

theorem static_store_other_program {α : Type} (codec : Codec α) (route : Route) (view : View)
    (d : Dimensions) (pid owner : Nat) (values : Tile α) (m : Heap)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) (different : owner ≠ pid) :
    load codec (storeTile codec route view d.width pid (columnIndex pid) values m)
      (view.address (route.components * columnIndex owner j + k)) =
      load codec m (view.address (route.components * columnIndex owner j + k)) := by
  apply storeTile_load_outside
  intro i hi l hl
  simpa only [timeIndex, Nat.zero_mul, Nat.zero_add] using
    component_other_program route d owner pid 0 0 j i hj hi k l hk hl different

theorem backwardProgram_other_input {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid owner t : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (outputs : OutputSeparation views) (ht : t < d.length)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) (different : owner ≠ pid) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradInput.address (route.components * timeIndex d t owner j + k)) =
      load codec m (views.gradInput.address (route.components * timeIndex d t owner j + k)) := by
  have frame := backwardTerminal_frame codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m) views.gradInput.backing
    (static_fits route views.gradInitial d.width pid sizes.gradInitial)
    (static_fits route views.gradTransition d.width pid sizes.gradTransition)
    outputs.initial_input outputs.transition_input
  have bound : route.components * timeIndex d t owner j + k < views.gradInput.words := by
    have fit := time_fits route views.gradInput d t owner sizes.gradInput ht j hj
    have hc := component_member route k hk
    omega
  change load codec (backwardTerminal codec route ops d views pid
    (backwardPrefix codec route graph ops d views pid d.length m)).heap _ = _
  rw [Frame.load codec views.gradInput frame _ bound]
  exact backwardPrefix_other_program codec route graph ops d views pid d.length owner t m j hj k hk different

theorem backwardProgram_other_initial {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid owner : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (outputs : OutputSeparation views)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) (different : owner ≠ pid) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradInitial.address (route.components * columnIndex owner j + k)) =
      load codec m (views.gradInitial.address (route.components * columnIndex owner j + k)) := by
  have fit := static_fits route views.gradInitial d.width owner sizes.gradInitial j hj
  have hc := component_member route k hk
  have bound : route.components * columnIndex owner j + k < views.gradInitial.words := by omega
  have hprefix := backwardPrefix_frame codec route graph ops d views pid d.length m views.gradInitial.backing
    (sizes.fits pid).gradInput outputs.initial_input.symm
  have last := storeTile_frame codec route views.gradTransition d.width pid (columnIndex pid)
    (fun j => terminalGradient route ops ((backwardPrefix codec route graph ops d views pid d.length m).registers j).transition)
    (storeTile codec route views.gradInitial d.width pid (columnIndex pid)
      (fun j => terminalGradient route ops ((backwardPrefix codec route graph ops d views pid d.length m).registers j).carry)
      (backwardPrefix codec route graph ops d views pid d.length m).heap)
    views.gradInitial.backing (static_fits route views.gradTransition d.width pid sizes.gradTransition) outputs.transition_initial
  change load codec (storeTile codec route views.gradTransition d.width pid (columnIndex pid) _
    (storeTile codec route views.gradInitial d.width pid (columnIndex pid) _
      (backwardPrefix codec route graph ops d views pid d.length m).heap)) _ = _
  refine (Frame.load codec views.gradInitial last _ bound).trans ?_
  rw [static_store_other_program codec route views.gradInitial d pid owner _ _ j hj k hk different,
    Frame.load codec views.gradInitial hprefix _ bound]

theorem backwardProgram_other_transition {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : BackwardViews) (pid owner : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (outputs : OutputSeparation views)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) (different : owner ≠ pid) :
    load codec (backwardProgram codec route graph ops d views pid m).heap
      (views.gradTransition.address (route.components * columnIndex owner j + k)) =
      load codec m (views.gradTransition.address (route.components * columnIndex owner j + k)) := by
  have fit := static_fits route views.gradTransition d.width owner sizes.gradTransition j hj
  have hc := component_member route k hk
  have bound : route.components * columnIndex owner j + k < views.gradTransition.words := by omega
  have hprefix := backwardPrefix_frame codec route graph ops d views pid d.length m views.gradTransition.backing
    (sizes.fits pid).gradInput outputs.transition_input.symm
  have middle := storeTile_frame codec route views.gradInitial d.width pid (columnIndex pid)
    (fun j => terminalGradient route ops ((backwardPrefix codec route graph ops d views pid d.length m).registers j).carry)
    (backwardPrefix codec route graph ops d views pid d.length m).heap views.gradTransition.backing
    (static_fits route views.gradInitial d.width pid sizes.gradInitial) outputs.transition_initial.symm
  change load codec (storeTile codec route views.gradTransition d.width pid (columnIndex pid) _
    (storeTile codec route views.gradInitial d.width pid (columnIndex pid) _
      (backwardPrefix codec route graph ops d views pid d.length m).heap)) _ = _
  rw [static_store_other_program codec route views.gradTransition d pid owner _ _ j hj k hk different]
  exact (Frame.load codec views.gradTransition middle _ bound).trans
    (Frame.load codec views.gradTransition hprefix _ bound)

/-- Generic fold lemma, instantiated below with derived task-local execution
theorems. Its hypotheses are not an external compiler or runtime contract. -/
theorem owned_grid_readback {State α : Type} (step : Nat → State → State)
    (observe value : State → α) (owner : Nat)
    (atOwner : ∀ s, observe (step owner s) = value s)
    (elsewhere : ∀ pid, owner ≠ pid → ∀ s, observe (step pid s) = observe s)
    (stable : ∀ pid s, value (step pid s) = value s)
    (order : List Nat) (member : owner ∈ order) (s : State) :
    observe (order.foldl (fun current pid => step pid current) s) = value s := by
  have keep (ids : List Nat) (missing : owner ∉ ids) (s : State) :
      observe (ids.foldl (fun current pid => step pid current) s) = observe s := by
    induction ids generalizing s with
    | nil => rfl
    | cons pid rest ih =>
      simp only [List.mem_cons, not_or] at missing
      rw [List.foldl_cons, ih missing.2, elsewhere pid missing.1]
  induction order generalizing s with
  | nil => simp at member
  | cons pid rest ih =>
    rw [List.foldl_cons]
    by_cases hin : owner ∈ rest
    · rw [ih hin, stable]
    · have he : owner = pid := (List.mem_cons.mp member).resolve_right hin
      subst pid
      rw [keep rest hin, atOwner]

theorem backwardGrid_input {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (order : List Nat) (owner : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardAllFresh views) (member : owner ∈ order)
    (i : Nat) (hi : i < d.length) (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardGrid codec route graph ops d views order m)
      (views.gradInput.address (route.components * timeIndex d (d.length - 1 - i) owner j + k)) =
      getComponent (backwardInputValues codec route graph ops d views owner m i j) k := by
  apply owned_grid_readback (fun pid heap => (backwardProgram codec route graph ops d views pid heap).heap)
    (fun heap => load codec heap (views.gradInput.address (route.components * timeIndex d (d.length - 1 - i) owner j + k)))
    (fun heap => getComponent (backwardInputValues codec route graph ops d views owner heap i j) k) owner
    _ _ _ order member m
  · intro heap
    exact backwardProgram_input codec route graph ops d views owner heap sizes fresh.prefix fresh.outputs i hi j hj k hk
  · intro pid different heap
    exact backwardProgram_other_input codec route graph ops d views pid owner _ heap sizes fresh.outputs (by omega) j hj k hk different
  · intro pid heap
    dsimp only
    rw [backwardInputValues_frame codec route graph ops d views owner i heap _ sizes
      (backwardProgram_read_frame codec route graph ops d views pid heap sizes fresh) hi]

theorem backwardGrid_initial {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (order : List Nat) (owner : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardAllFresh views) (member : owner ∈ order)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardGrid codec route graph ops d views order m)
      (views.gradInitial.address (route.components * columnIndex owner j + k)) =
      getComponent (terminalGradient route ops (backwardValues codec route graph ops d views owner m d.length j).carry) k := by
  apply owned_grid_readback (fun pid heap => (backwardProgram codec route graph ops d views pid heap).heap)
    (fun heap => load codec heap (views.gradInitial.address (route.components * columnIndex owner j + k)))
    (fun heap => getComponent (terminalGradient route ops (backwardValues codec route graph ops d views owner heap d.length j).carry) k) owner
    _ _ _ order member m
  · intro heap
    exact backwardProgram_initial codec route graph ops d views owner heap sizes fresh.prefix fresh.outputs j hj k hk
  · intro pid different heap
    exact backwardProgram_other_initial codec route graph ops d views pid owner heap sizes fresh.outputs j hj k hk different
  · intro pid heap
    dsimp only
    rw [backwardValues_frame codec route graph ops d views owner d.length heap _ sizes
      (backwardProgram_read_frame codec route graph ops d views pid heap sizes fresh) le_rfl]

theorem backwardGrid_transition {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : BackwardViews) (order : List Nat) (owner : Nat) (m : Heap)
    (sizes : BackwardSizes route d views) (fresh : BackwardAllFresh views) (member : owner ∈ order)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route) :
    load codec (backwardGrid codec route graph ops d views order m)
      (views.gradTransition.address (route.components * columnIndex owner j + k)) =
      getComponent (terminalGradient route ops (backwardValues codec route graph ops d views owner m d.length j).transition) k := by
  apply owned_grid_readback (fun pid heap => (backwardProgram codec route graph ops d views pid heap).heap)
    (fun heap => load codec heap (views.gradTransition.address (route.components * columnIndex owner j + k)))
    (fun heap => getComponent (terminalGradient route ops (backwardValues codec route graph ops d views owner heap d.length j).transition) k) owner
    _ _ _ order member m
  · intro heap
    exact backwardProgram_transition codec route graph ops d views owner heap sizes fresh.prefix j hj k hk
  · intro pid different heap
    exact backwardProgram_other_transition codec route graph ops d views pid owner heap sizes fresh.outputs j hj k hk different
  · intro pid heap
    dsimp only
    rw [backwardValues_frame codec route graph ops d views owner d.length heap _ sizes
      (backwardProgram_read_frame codec route graph ops d views pid heap sizes fresh) le_rfl]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
