import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ForwardExecution

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 4000000

structure ForwardSizes (route : Route) (d : Dimensions) (views : ForwardViews) : Prop where
  initial : views.initial.words = route.components * d.width
  transition : views.transition.words = route.components * d.dim
  input : views.input.words = route.components * (d.length * d.width)
  output : views.output.words = route.components * (d.length * d.width)

structure ForwardFresh (views : ForwardViews) : Prop where
  initial : views.output.backing.disjoint views.initial.backing
  input : views.output.backing.disjoint views.input.backing
  transition : views.output.backing.disjoint views.transition.backing

structure ForwardReadFrame (views : ForwardViews) (before after : Heap) : Prop where
  initial : Frame before after views.initial.backing
  input : Frame before after views.input.backing
  transition : Frame before after views.transition.backing

theorem forwardProgram_read_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap)
    (sizes : ForwardSizes route d views) (fresh : ForwardFresh views) :
    ForwardReadFrame views m (forwardProgram codec route graph ops d views pid m).heap := by
  have fits := fun t ht => time_fits route views.output d t pid sizes.output ht
  exact ⟨forwardPrefix_frame codec route graph ops d views pid d.length m _ fits fresh.initial,
    forwardPrefix_frame codec route graph ops d views pid d.length m _ fits fresh.input,
    forwardPrefix_frame codec route graph ops d views pid d.length m _ fits fresh.transition⟩

theorem forwardValues_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid steps : Nat) (before after : Heap)
    (sizes : ForwardSizes route d views) (frame : ForwardReadFrame views before after) (bound : steps ≤ d.length) :
    forwardValues codec route graph ops d views pid after steps =
      forwardValues codec route graph ops d views pid before steps := by
  induction steps with
  | zero =>
    exact readTile_frame codec route ops views.initial d.width pid (columnIndex pid) before after
      (static_fits route views.initial d.width pid sizes.initial) frame.initial
  | succ steps ih =>
    have ih := ih (by omega)
    have transition := readTile_frame codec route ops views.transition d.width pid (transitionIndex d pid) before after
      (transition_fits route views.transition d pid sizes.transition) frame.transition
    have input := readTile_frame codec route ops views.input d.width pid (timeIndex d steps pid) before after
      (time_fits route views.input d steps pid sizes.input (by omega)) frame.input
    simp only [forwardValues, ih, transition, input]

theorem component_other_program (route : Route) (d : Dimensions) (owner pid t u : Nat)
    (j i : Lane) (hj : active d.width owner j) (hi : active d.width pid i)
    (k l : Nat) (hk : k ∈ componentList route) (hl : l ∈ componentList route) (different : owner ≠ pid) :
    route.components * timeIndex d t owner j + k ≠ route.components * timeIndex d u pid i + l := by
  intro same
  have scalar := (component_index_unique route _ _ k l
    (component_member route k hk) (component_member route l hl) same).1
  have coordinate := (NativeIndex.temporal_unique d.width t u
    (columnIndex owner j) (columnIndex pid i) hj hi scalar).2
  exact different (NativeIndex.column_unique owner pid j.val i.val j.isLt i.isLt coordinate).1

theorem forwardPrefix_other_program {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid steps owner t : Nat) (m : Heap)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route)
    (different : owner ≠ pid) :
    load codec (forwardPrefix codec route graph ops d views pid steps m).heap
      (views.output.address (route.components * timeIndex d t owner j + k)) =
      load codec m (views.output.address (route.components * timeIndex d t owner j + k)) := by
  induction steps with
  | zero => rfl
  | succ steps ih =>
    rw [forwardPrefix_heap_step, storeTile_load_outside]
    · exact ih
    · intro i hi l hl
      exact component_other_program route d owner pid t steps j i hj hi k l hk hl different

theorem forwardGrid_other_program {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (order : List Nat) (owner t : Nat) (m : Heap)
    (j : Lane) (hj : active d.width owner j) (k : Nat) (hk : k ∈ componentList route)
    (missing : owner ∉ order) :
    load codec (forwardGrid codec route graph ops d views order m)
      (views.output.address (route.components * timeIndex d t owner j + k)) =
      load codec m (views.output.address (route.components * timeIndex d t owner j + k)) := by
  induction order generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    change load codec (forwardGrid codec route graph ops d views rest
      (forwardProgram codec route graph ops d views pid m).heap) _ = _
    rw [ih _ (by simp only [List.mem_cons, not_or] at missing; exact missing.2)]
    exact forwardPrefix_other_program codec route graph ops d views pid d.length owner t m j hj k hk
      (by simp only [List.mem_cons, not_or] at missing; exact missing.1)

/-- An arbitrary order containing this coordinate's owning program returns
the derived value. No per-program result or success is assumed. Repeated
programs are harmless because all operand backings are preserved. -/
theorem forwardGrid_output {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (order : List Nat) (owner t : Nat) (m : Heap)
    (sizes : ForwardSizes route d views) (fresh : ForwardFresh views)
    (member : owner ∈ order) (ht : t < d.length) (j : Lane) (hj : active d.width owner j)
    (k : Nat) (hk : k ∈ componentList route) :
    load codec (forwardGrid codec route graph ops d views order m)
      (views.output.address (route.components * timeIndex d t owner j + k)) =
      getComponent (forwardValues codec route graph ops d views owner m (t + 1) j) k := by
  induction order generalizing m with
  | nil => simp at member
  | cons pid rest ih =>
    change load codec (forwardGrid codec route graph ops d views rest
      (forwardProgram codec route graph ops d views pid m).heap) _ = _
    by_cases inRest : owner ∈ rest
    · rw [ih _ inRest]
      rw [forwardValues_frame codec route graph ops d views owner (t + 1) m
        (forwardProgram codec route graph ops d views pid m).heap sizes
        (forwardProgram_read_frame codec route graph ops d views pid m sizes fresh) (by omega)]
    · have same : owner = pid := (List.mem_cons.mp member).resolve_right inRest
      subst pid
      rw [forwardGrid_other_program codec route graph ops d views rest owner t _ j hj k hk inRest]
      exact forwardProgram_output codec route graph ops d views owner m sizes.input sizes.output fresh.input
        t ht j hj k hk

theorem forwardGrid_full_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (order : List Nat) (m : Heap)
    (outputSize : views.output.words = route.components * (d.length * d.width))
    (p : Nat) (outside : p < views.output.pointer ∨ views.output.pointer + 4 * views.output.words ≤ p) :
    forwardGrid codec route graph ops d views order m p = m p := by
  induction order generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    change forwardGrid codec route graph ops d views rest (forwardProgram codec route graph ops d views pid m).heap p = _
    rw [ih]
    exact forwardPrefix_full_frame codec route graph ops d views pid d.length m
      (fun t ht => time_fits route views.output d t pid outputSize ht) p outside

theorem forwardGrid_caller_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (order : List Nat) (m : Heap)
    (sizes : ForwardSizes route d views) (caller : Backing) (fresh : views.output.backing.disjoint caller) :
    Frame m (forwardGrid codec route graph ops d views order m) caller := by
  induction order generalizing m with
  | nil => exact Frame.refl _ _
  | cons pid rest ih =>
    exact (forwardPrefix_frame codec route graph ops d views pid d.length m caller
      (fun t ht => time_fits route views.output d t pid sizes.output ht) fresh).trans (ih _)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
