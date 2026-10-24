import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Execution

set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Execution
open Kernels ByteMemory

theorem components_pos (route : Route) : 0 < route.components := by
  cases route <;> decide

theorem component_index_unique (route : Route) (a b k l : Nat)
    (hk : k < route.components) (hl : l < route.components)
    (he : route.components * a + k = route.components * b + l) : a = b ∧ k = l := by
  cases route <;> simp only [Route.components] at * <;> omega

theorem timeIndex_injective (d : Dimensions) (t pid : Nat) (j k : Lane)
    (same : timeIndex d t pid j = timeIndex d t pid k) : j = k := by
  apply Fin.ext
  simp only [timeIndex, columnIndex, NativeIndex.column] at same
  omega

theorem timeIndex_disjoint (d : Dimensions) (t u p q : Nat) (j k : Lane)
    (hj : active d.width p j) (hk : active d.width q k) (different : t ≠ u) :
    timeIndex d t p j ≠ timeIndex d u q k := by
  intro same
  exact different (NativeIndex.temporal_unique d.width t u (columnIndex p j)
    (columnIndex q k) hj hk same).1

theorem static_fits (route : Route) (view : View) (n pid : Nat)
    (size : view.words = route.components * n) :
    TileFits route view n pid (columnIndex pid) := by
  intro j hj
  rw [size]
  have hc := components_pos route
  change NativeIndex.column pid j.val < n at hj
  unfold columnIndex
  nlinarith

theorem time_fits (route : Route) (view : View) (d : Dimensions) (t pid : Nat)
    (size : view.words = route.components * (d.length * d.width)) (ht : t < d.length) :
    TileFits route view d.width pid (timeIndex d t pid) := by
  intro j hj
  rw [size]
  have hc := components_pos route
  have h := NativeIndex.active_offset d.length d.width t (columnIndex pid j) ht hj
  change route.components * (t * d.width + columnIndex pid j) + route.components ≤ _
  nlinarith

theorem transition_fits (route : Route) (view : View) (d : Dimensions) (pid : Nat)
    (size : view.words = route.components * d.dim) :
    TileFits route view d.width pid (transitionIndex d pid) := by
  intro j hj
  rw [size]
  have hd := (NativeIndex.active_dimension d.batch d.dim (columnIndex pid j) hj).2
  have h := Nat.mod_lt (columnIndex pid j) hd
  have hc := components_pos route
  unfold transitionIndex
  nlinarith

theorem storeTile_read {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (m : Heap)
    (j : Lane) (activeJ : active n pid j) (k : Nat) (componentK : k ∈ componentList route)
    (unique : ∀ a b, active n pid a → active n pid b → index a = index b → a = b) :
    load codec (storeTile codec route view n pid index values m)
      (view.address (route.components * index j + k)) = getComponent (values j) k := by
  apply stores_readback
  · apply List.mem_flatMap.mpr
    refine ⟨k, componentK, ?_⟩
    exact List.mem_map.mpr ⟨j, (selected_member n pid j).2 activeJ, rfl⟩
  · intro event he
    obtain ⟨l, hl, he⟩ := List.mem_flatMap.mp he
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
    have activeI := (selected_member n pid i).1 hi
    by_cases same : route.components * index i + l = route.components * index j + k
    · have eqs := component_index_unique route (index i) (index j) l k
        (component_member route l hl) (component_member route k componentK) same
      have ij := unique i j activeI activeJ eqs.1
      subst i
      rcases eqs with ⟨_, rfl⟩
      exact Or.inl rfl
    · exact Or.inr (view.word_disjoint _ _ (Ne.symm same))

theorem storeTile_load_outside {α : Type} (codec : Codec α) (route : Route) (view : View)
    (n pid : Nat) (index : Lane → Nat) (values : Tile α) (m : Heap) (i : Nat)
    (outside : ∀ j, active n pid j → ∀ k ∈ componentList route,
      i ≠ route.components * index j + k) :
    load codec (storeTile codec route view n pid index values m) (view.address i) =
      load codec m (view.address i) := by
  apply stores_load_other
  intro event he
  obtain ⟨k, hk, he⟩ := List.mem_flatMap.mp he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.mp he
  exact view.word_disjoint _ _ (outside j ((selected_member n pid j).1 hj) k hk)

theorem storeTile_previous_time {α : Type} (codec : Codec α) (route : Route) (view : View)
    (d : Dimensions) (pid next prior : Nat) (values : Tile α) (m : Heap)
    (j : Lane) (hj : active d.width pid j) (k : Nat) (hk : k ∈ componentList route)
    (different : prior ≠ next) :
    load codec (storeTile codec route view d.width pid (timeIndex d next pid) values m)
      (view.address (route.components * timeIndex d prior pid j + k)) =
      load codec m (view.address (route.components * timeIndex d prior pid j + k)) := by
  apply storeTile_load_outside
  intro i hi l hl same
  have eqs := component_index_unique route (timeIndex d prior pid j) (timeIndex d next pid i) k l
    (component_member route k hk) (component_member route l hl) same
  exact timeIndex_disjoint d prior next pid pid j i hj hi different eqs.1

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Execution
