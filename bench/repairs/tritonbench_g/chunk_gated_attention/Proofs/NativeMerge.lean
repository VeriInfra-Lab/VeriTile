import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeMergeLemmas

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

def InnerInputsSame (before after : NativeGrid.State) : Prop :=
  ∀ (r : RegionName) off,r≠"h" → r≠"ht" → after.mem r off=before.mem r off
def PrefixInputsSame (before after : NativeGrid.State) : Prop :=
  ∀ off,after.mem "s" off=before.mem "s" off

theorem inner_merge_inputs (c : HostFP.Inner) (p : LaunchPlan.Item) (before current : NativeGrid.State)
    (same : InnerInputsSame before current) :
    InnerInputsSame before (NativeGrid.merge (NativeGrid.innerReference c) (NativeGrid.innerOwns c) p current) := by
  intro r off hr ht
  rw [merge_frame _ _ _ _ r off]
  · exact same r off hr ht
  · rintro ⟨i,own⟩
    have hit := (inner_owns_hits c p i current r off).mp own
    rw [NativeStores.inner_hits] at hit
    rcases hit with ⟨_,_,region,_⟩|⟨_,region,_⟩
    · exact hr region
    · exact ht region

theorem prefix_merge_inputs (c : HostFP.Cumulative) (p : LaunchPlan.Item) (before current : NativeGrid.State)
    (same : PrefixInputsSame before current) :
    PrefixInputsSame before (NativeGrid.merge (NativeGrid.prefixReference c) (NativeGrid.prefixOwns c) p current) := by
  intro off
  rw [merge_frame _ _ _ _ "s" off]
  · exact same off
  · rintro ⟨i,own⟩
    have hit := (prefix_owns_hits c p i current "s" off).mp own
    rw [NativeStores.prefix_hits] at hit
    have clash := hit.2.1
    contradiction

/-- The original ordinary grid result equals the full finite native store
family. Arbitrary native configs, source dummies, register and undef values
remain admitted. Only proved ownership and read-role frames are used. -/
theorem inner_merge_serial (c : HostFP.Inner) (before : NativeGrid.State) :
    (NativeGrid.innerResult c before).mem=
      NativeStores.apply ((allJobs c.plans).flatMap fun j=>innerStores c j before) before.mem := by
  funext r off
  by_cases hit : NativeStores.Hits ((allJobs c.plans).flatMap fun j=>innerStores c j before) r off
  · obtain ⟨selected,listed,selectedHit⟩ := (hits_flatMap _ _ r off).mp hit
    rw [apply_flatMap_owned _ _ selected listed r off selectedHit
      (fun j _ hj=>inner_disjoint c before before j selected r off hj selectedHit)]
    refine batches_owned_value (NativeGrid.innerReference c) (NativeGrid.innerOwns c) c.plans
      (InnerInputsSame before) (fun p _ s=>inner_merge_inputs c p before s) r off _ ?_ ?_ before ?_
    · intro p hp current same lane own
      obtain ⟨index,bound,rfl⟩ := List.getElem_of_mem hp
      let job : Job c.plans := ⟨⟨index,bound⟩,lane⟩
      have jobHit : NativeStores.Hits (innerStores c job current) r off :=
        (inner_owns_hits c _ lane current r off).mp own
      have equal := inner_disjoint c current before job selected r off jobHit selectedHit
      rw [NativeGrid.innerReference,NativeStores.inner_memory]
      change NativeStores.apply (innerStores c job current) current.mem r off=_
      rw [innerStores_same c job current before same,equal]
      exact NativeStores.at_hit _ _ _ r off selectedHit
    · exact ⟨plan selected,plan_mem selected,selected.2,
        (inner_owns_hits c _ selected.2 before r off).mpr selectedHit⟩
    · exact fun _ _ _ _=>rfl
  · rw [NativeStores.frame _ _ r off hit]
    apply RawNative.frame
    rintro ⟨p,hp,lane,own⟩
    obtain ⟨index,bound,rfl⟩ := List.getElem_of_mem hp
    let job : Job c.plans := ⟨⟨index,bound⟩,lane⟩
    apply hit
    exact (hits_flatMap _ _ r off).mpr ⟨job,by simp [allJobs],
      (inner_owns_hits c _ lane before r off).mp own⟩

theorem prefix_merge_serial (c : HostFP.Cumulative) (before : NativeGrid.State) :
    (NativeGrid.prefixResult c before).mem=
      NativeStores.apply ((allJobs c.plans).flatMap fun j=>prefixStores c j before) before.mem := by
  funext r off
  by_cases hit : NativeStores.Hits ((allJobs c.plans).flatMap fun j=>prefixStores c j before) r off
  · obtain ⟨selected,listed,selectedHit⟩ := (hits_flatMap _ _ r off).mp hit
    rw [apply_flatMap_owned _ _ selected listed r off selectedHit
      (fun j _ hj=>prefix_disjoint c before before j selected r off hj selectedHit)]
    refine batches_owned_value (NativeGrid.prefixReference c) (NativeGrid.prefixOwns c) c.plans
      (PrefixInputsSame before) (fun p _ s=>prefix_merge_inputs c p before s) r off _ ?_ ?_ before ?_
    · intro p hp current same lane own
      obtain ⟨index,bound,rfl⟩ := List.getElem_of_mem hp
      let job : Job c.plans := ⟨⟨index,bound⟩,lane⟩
      have jobHit : NativeStores.Hits (prefixStores c job current) r off :=
        (prefix_owns_hits c _ lane current r off).mp own
      have equal := prefix_disjoint c current before job selected r off jobHit selectedHit
      rw [NativeGrid.prefixReference,NativeStores.prefix_memory]
      change NativeStores.apply (prefixStores c job current) current.mem r off=_
      rw [prefixStores_same c job current before same,equal]
      exact NativeStores.at_hit _ _ _ r off selectedHit
    · exact ⟨plan selected,plan_mem selected,selected.2,
        (prefix_owns_hits c _ selected.2 before r off).mpr selectedHit⟩
    · exact fun _=>rfl
  · rw [NativeStores.frame _ _ r off hit]
    apply RawNative.frame
    rintro ⟨p,hp,lane,own⟩
    obtain ⟨index,bound,rfl⟩ := List.getElem_of_mem hp
    let job : Job c.plans := ⟨⟨index,bound⟩,lane⟩
    apply hit
    exact (hits_flatMap _ _ r off).mpr ⟨job,by simp [allJobs],
      (prefix_owns_hits c _ lane before r off).mp own⟩

theorem inner_schedule_original (c : HostFP.Inner) (before : NativeGrid.State) :
    ∃ after,runSchedule ((allJobs c.plans).map (invocation (NativeGrid.innerSource c))) before=some after ∧
      after.mem=(NativeGrid.innerResult c before).mem := by
  obtain ⟨after,ran,memory⟩ := inner_run c (allJobs c.plans) before
  exact ⟨after,ran,memory.trans (inner_merge_serial c before).symm⟩

theorem prefix_schedule_original (c : HostFP.Cumulative) (before : NativeGrid.State) :
    ∃ after,runSchedule ((allJobs c.plans).map (invocation (NativeGrid.prefixSource c))) before=some after ∧
      after.mem=(NativeGrid.prefixResult c before).mem := by
  obtain ⟨after,ran,memory⟩ := prefix_run c (allJobs c.plans) before
  exact ⟨after,ran,memory.trans (prefix_merge_serial c before).symm⟩

#axiomsClean inner_merge_serial
#axiomsClean prefix_merge_serial
#axiomsClean inner_schedule_original
#axiomsClean prefix_schedule_original
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
