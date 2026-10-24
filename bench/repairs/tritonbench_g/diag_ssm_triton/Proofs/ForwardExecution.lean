import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.TileMemory

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory
set_option maxHeartbeats 3000000

/-- Actual forward register recurrence, parameterized by the exact primitive
graph. The later numerical layer relates it to the independent Real target. -/
def forwardValues {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap) : Nat → Tile α
  | 0 => readTile codec route ops views.initial d.width pid (columnIndex pid) m
  | k + 1 => fun j => Kernels.forward route graph ops
      (forwardValues codec route graph ops d views pid m k j)
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m j)
      (readTile codec route ops views.input d.width pid (timeIndex d k pid) m j)

def forwardPrefix {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid k : Nat) (m : Heap) : Machine α :=
  forwardLoop codec route graph ops d views pid
    (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m) k
    (forwardInitial codec route ops d views pid m)

theorem forwardPrefix_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid k : Nat) (m : Heap) (caller : Backing)
    (fits : ∀ t < k, TileFits route views.output d.width pid (timeIndex d t pid))
    (fresh : views.output.backing.disjoint caller) :
    Frame m (forwardPrefix codec route graph ops d views pid k m).heap caller :=
  forwardLoop_frame codec route graph ops d views pid _ k _ caller fits fresh

theorem forwardPrefix_values {α : Type} (codec : Codec α) (route : Route) (graph : Graph) (ops : Operations α)
    (d : Dimensions) (views : ForwardViews) (pid k : Nat) (m : Heap)
    (inputFits : ∀ t < k, TileFits route views.input d.width pid (timeIndex d t pid))
    (outputFits : ∀ t < k, TileFits route views.output d.width pid (timeIndex d t pid))
    (fresh : views.output.backing.disjoint views.input.backing) :
    (forwardPrefix codec route graph ops d views pid k m).registers =
      forwardValues codec route graph ops d views pid m k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have ih := ih (by intro t ht; exact inputFits t (by omega))
      (by intro t ht; exact outputFits t (by omega))
    have frame := forwardPrefix_frame codec route graph ops d views pid k m views.input.backing
      (by intro t ht; exact outputFits t (by omega)) fresh
    have reads := readTile_frame codec route ops views.input d.width pid (timeIndex d k pid) m
      (forwardPrefix codec route graph ops d views pid k m).heap (inputFits k (by omega)) frame
    change (fun j => Kernels.forward route graph ops
      ((forwardPrefix codec route graph ops d views pid k m).registers j)
      (readTile codec route ops views.transition d.width pid (transitionIndex d pid) m j)
      (readTile codec route ops views.input d.width pid (timeIndex d k pid)
        (forwardPrefix codec route graph ops d views pid k m).heap j)) = _
    rw [ih, reads]
    rfl

theorem forwardPrefix_heap_step {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid k : Nat) (m : Heap) :
    (forwardPrefix codec route graph ops d views pid (k + 1) m).heap =
      storeTile codec route views.output d.width pid (timeIndex d k pid)
        (forwardPrefix codec route graph ops d views pid (k + 1) m).registers
        (forwardPrefix codec route graph ops d views pid k m).heap := rfl

/-- Readback comes from source stores into evolving physical memory. Neither
zero initialization nor a desired output equality is an input premise. -/
theorem forwardPrefix_output {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid steps : Nat) (m : Heap)
    (inputFits : ∀ t < steps, TileFits route views.input d.width pid (timeIndex d t pid))
    (outputFits : ∀ t < steps, TileFits route views.output d.width pid (timeIndex d t pid))
    (fresh : views.output.backing.disjoint views.input.backing)
    (t : Nat) (ht : t < steps) (j : Lane) (hj : active d.width pid j)
    (k : Nat) (hk : k ∈ componentList route) :
    load codec (forwardPrefix codec route graph ops d views pid steps m).heap
      (views.output.address (route.components * timeIndex d t pid j + k)) =
      getComponent (forwardValues codec route graph ops d views pid m (t + 1) j) k := by
  induction steps with
  | zero => omega
  | succ steps ih =>
    rw [forwardPrefix_heap_step]
    by_cases equalTime : t = steps
    · subst t
      rw [storeTile_read codec route views.output d.width pid (timeIndex d steps pid) _ _ j hj k hk
        (by intro a b _ _ same; exact timeIndex_injective d steps pid a b same)]
      rw [forwardPrefix_values codec route graph ops d views pid (steps + 1) m inputFits outputFits fresh]
    · rw [storeTile_previous_time codec route views.output d pid steps t _ _ j hj k hk equalTime]
      exact ih (by intro q hq; exact inputFits q (by omega))
        (by intro q hq; exact outputFits q (by omega)) (by omega)

theorem storeTile_outside_view {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (m : Heap)
    (fits : TileFits route view n pid index) (p : Nat)
    (outside : p < view.pointer ∨ view.pointer + 4 * view.words ≤ p) :
    storeTile codec route view n pid index values m p = m p := by
  apply stores_frame
  intro event he
  obtain ⟨k, hk, he⟩ := List.mem_flatMap.mp he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
  have hk := component_member route k hk
  have hf := fits j ((selected_member n pid j).1 hj)
  simp only [View.address]
  omega

/-- Prefix/suffix allocation padding and every byte outside the actual output
view are preserved, not merely the logical input tensors. -/
theorem forwardPrefix_full_frame {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid steps : Nat) (m : Heap)
    (outputFits : ∀ t < steps, TileFits route views.output d.width pid (timeIndex d t pid))
    (p : Nat) (outside : p < views.output.pointer ∨ views.output.pointer + 4 * views.output.words ≤ p) :
    (forwardPrefix codec route graph ops d views pid steps m).heap p = m p := by
  induction steps with
  | zero => rfl
  | succ steps ih =>
    rw [forwardPrefix_heap_step, storeTile_outside_view codec route views.output d.width pid
      (timeIndex d steps pid) _ _ (outputFits steps (by omega)) p outside]
    exact ih (by intro t ht; exact outputFits t (by omega))

theorem forwardProgram_output {α : Type} (codec : Codec α) (route : Route) (graph : Graph)
    (ops : Operations α) (d : Dimensions) (views : ForwardViews) (pid : Nat) (m : Heap)
    (inputSize : views.input.words = route.components * (d.length * d.width))
    (outputSize : views.output.words = route.components * (d.length * d.width))
    (fresh : views.output.backing.disjoint views.input.backing)
    (t : Nat) (ht : t < d.length) (j : Lane) (hj : active d.width pid j)
    (k : Nat) (hk : k ∈ componentList route) :
    load codec (forwardProgram codec route graph ops d views pid m).heap
      (views.output.address (route.components * timeIndex d t pid j + k)) =
      getComponent (forwardValues codec route graph ops d views pid m (t + 1) j) k :=
  forwardPrefix_output codec route graph ops d views pid d.length m
    (fun t ht => time_fits route views.input d t pid inputSize ht)
    (fun t ht => time_fits route views.output d t pid outputSize ht) fresh t ht j hj k hk

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
