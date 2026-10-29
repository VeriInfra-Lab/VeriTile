import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeDisjoint
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.RawNative

/-! Task-local list/merge lemmas connecting the original ordinary grids to
the proved finite-store schedule. They introduce no alternate executor. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

theorem apply_flatMap_owned {J : Type} (jobs : List J) (writes : J → List NativeStores.Store)
    (selected : J) (listed : selected∈jobs) (r : RegionName) (off : Nat)
    (hit : NativeStores.Hits (writes selected) r off)
    (unique : ∀ j∈jobs,NativeStores.Hits (writes j) r off → j=selected)
    (memory : NativeStores.Memory) :
    NativeStores.apply (jobs.flatMap writes) memory r off=
      NativeStores.apply (writes selected) memory r off := by
  induction jobs generalizing memory with
  | nil => simp at listed
  | cons first rest ih =>
    simp only [List.flatMap_cons,NativeStores.apply_append]
    by_cases tail : selected∈rest
    · rw [ih tail (fun j hj=>unique j (List.mem_cons_of_mem first hj))]
      exact NativeStores.at_hit _ _ _ r off hit
    · have noTail : ¬ NativeStores.Hits (rest.flatMap writes) r off := by
        rw [hits_flatMap]
        rintro ⟨j,hj,hwrite⟩
        have same := unique j (List.mem_cons_of_mem first hj) hwrite
        exact tail (same ▸ hj)
      rw [NativeStores.frame _ _ r off noTail]
      have same := (List.mem_cons.mp listed).resolve_right tail
      subst first
      rfl

theorem merge_frame (reference : Nat → NativeGrid.State → NativeGrid.State)
    (owns : (p : LaunchPlan.Item) → HostFP.Ownership p.size) (p : LaunchPlan.Item)
    (s : NativeGrid.State) (r : RegionName) (off : Nat) (outside : ¬∃i,owns p i r off) :
    (NativeGrid.merge reference owns p s).mem r off=s.mem r off := by
  simp only [NativeGrid.merge,dif_neg outside]

/-- Once every possible owner supplies the same selected native cell, the
original successive ordinary-grid merges retain that value. The invariant
records only preserved input memory, and is discharged from exact frames. -/
theorem batches_owned_value (reference : Nat → NativeGrid.State → NativeGrid.State)
    (owns : (p : LaunchPlan.Item) → HostFP.Ownership p.size)
    (plans : List LaunchPlan.Item) (invariant : NativeGrid.State → Prop)
    (preserve : ∀ p∈plans,∀ s,invariant s → invariant (NativeGrid.merge reference owns p s))
    (r : RegionName) (off : Nat) (value : Cell NativeGraph.Expr)
    (owned : ∀ p∈plans,∀ s,invariant s → ∀ i,owns p i r off →
      (reference p.base (HostFP.gridState s p.size i.val)).mem r off=value)
    (existsOwner : RawNative.Writes owns plans r off)
    (s : NativeGrid.State) (initial : invariant s) :
    (NativeGrid.referenceBatches reference owns plans s).mem r off=value := by
  induction plans generalizing s with
  | nil => simp [RawNative.Writes] at existsOwner
  | cons p rest ih =>
    change (NativeGrid.referenceBatches reference owns rest (NativeGrid.merge reference owns p s)).mem r off=value
    by_cases tail : RawNative.Writes owns rest r off
    · exact ih (fun q hq=>preserve q (List.mem_cons_of_mem p hq))
        (fun q hq=>owned q (List.mem_cons_of_mem p hq)) tail _
        (preserve p List.mem_cons_self s initial)
    · rw [RawNative.frame reference owns rest _ r off tail]
      have head : ∃ i,owns p i r off := by
        obtain ⟨q,hq,i,hi⟩ := existsOwner
        rcases List.mem_cons.mp hq with same|member
        · subst q; exact ⟨i,hi⟩
        · exact False.elim (tail ⟨q,member,i,hi⟩)
      simp only [NativeGrid.merge,dif_pos head]
      exact owned p List.mem_cons_self s initial _ (Classical.choose_spec head)

theorem job_of_member (plans : List LaunchPlan.Item) (p : LaunchPlan.Item)
    (member : p∈plans) (lane : Fin p.size) :
    ∃ j : Job plans,plan j=p ∧ j.2.val=lane.val := by
  obtain ⟨index,bound,rfl⟩ := List.getElem_of_mem member
  exact ⟨⟨⟨index,bound⟩,lane⟩,rfl,rfl⟩

#axiomsClean apply_flatMap_owned
#axiomsClean batches_owned_value
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
