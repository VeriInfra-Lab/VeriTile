import bench.tritonbench_g.nested_loops_processing.Proofs.Execution

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical

theorem tile_copy_origin (count cols outShift inShift : Nat) (copy : Copy count)
    (member : copy∈tileCopies count cols outShift inShift) :
    ∃ lane:Lane,activeLane cols lane ∧ copy.destination.val=address cols outShift lane ∧
      copy.source=address cols inShift lane := by
  obtain ⟨lane,_,found⟩:=List.mem_filterMap.mp member
  unfold laneCopy at found
  split at found
  · rename_i h
    have eq:=Option.some.inj found
    subst copy
    exact ⟨lane,h.2,rfl,rfl⟩
  · simp at found

theorem tile_copy_same_source (count cols outShift inShift : Nat) (positive : 0<cols)
    (a b : Copy count) (ha : a∈tileCopies count cols outShift inShift)
    (hb : b∈tileCopies count cols outShift inShift) (same : a.destination.val=b.destination.val) :
    a.source=b.source := by
  obtain ⟨la,aa,da,sa⟩:=tile_copy_origin _ _ _ _ a ha
  obtain ⟨lb,ab,db,sb⟩:=tile_copy_origin _ _ _ _ b hb
  have lanes:=active_lane_injective cols outShift positive la lb aa ab (da.symm.trans (same.trans db))
  rw [sa,sb,lanes]

theorem tile_steps_commute (zero : α) (count cols outShift inShift : Nat) (positive : 0<cols)
    (input memory : Nat → α) (a b : Copy count)
    (ha : a∈tileCopies count cols outShift inShift) (hb : b∈tileCopies count cols outShift inShift) :
    wordStep zero count input (wordStep zero count input memory a) b=
      wordStep zero count input (wordStep zero count input memory b) a := by
  by_cases same:a.destination.val=b.destination.val
  · have source:=tile_copy_same_source _ _ _ _ positive a b ha hb same
    simp [wordStep,copyValue,same,source]
  · exact Function.update_comm same _ _ memory

theorem tile_order_independent (zero : α) (count cols outShift inShift : Nat) (positive : 0<cols)
    (input memory : Nat → α) (order : List (Copy count))
    (perm : order.Perm (tileCopies count cols outShift inShift)) :
    wordRun zero count input order memory=wordRun zero count input (tileCopies count cols outShift inShift) memory := by
  apply perm.foldl_eq'
  intro a ha b hb memory
  exact tile_steps_commute zero count cols outShift inShift positive input memory a b
    (perm.mem_iff.mp ha) (perm.mem_iff.mp hb)

/-- One legal linearization per source tl.store. Source statement order is
preserved; active lanes inside a statement may be in any order. -/
structure Schedule (count cols : Nat) where
  tile : Nat → List (Copy count)
  permutation : ∀ t,t<24 → (tile t).Perm (tileCopies count cols (2*t) (CopyMath.inputShift t))

def Schedule.order (schedule : Schedule count cols) : List (Copy count) :=
  (List.range 24).flatMap schedule.tile

def Schedule.canonical (count cols : Nat) : Schedule count cols :=
  ⟨fun t=>tileCopies count cols (2*t) (CopyMath.inputShift t),fun _ _=>List.Perm.refl _⟩

theorem scheduled_words (zero : α) (schedule : Schedule count cols) (positive : 0<cols)
    (input initial : Nat → α) :
    wordRun zero count input schedule.order initial=wordRun zero count input (copies count cols) initial := by
  have fold : ∀ ts : List Nat,(∀ t∈ts,t<24) → ∀ initial : Nat → α,
      wordRun zero count input (ts.flatMap schedule.tile) initial=
      wordRun zero count input (ts.flatMap (fun t=>tileCopies count cols (2*t) (CopyMath.inputShift t))) initial := by
    intro ts
    induction ts with
    | nil=>intro _ _;rfl
    | cons t ts ih=>
      intro bound initial
      simp only [List.flatMap_cons,wordRun,List.foldl_append]
      rw [show (schedule.tile t).foldl (wordStep zero count input) initial=
        (tileCopies count cols (2*t) (CopyMath.inputShift t)).foldl (wordStep zero count input) initial from
          tile_order_independent zero count cols (2*t) (CopyMath.inputShift t) positive input initial
            (schedule.tile t) (schedule.permutation t (bound t (by simp)))]
      exact ih (fun u hu=>bound u (by simp [hu])) _
  exact fold (List.range 24) (fun t ht=>List.mem_range.mp ht) initial

theorem execute_schedule_readback (api : API α Store β) (zero : α) (p : Placement count)
    (schedule : Schedule count cols) (positive : 0<cols) (memory : Store)
    (i : Nat) (bound : i<count) :
    api.read (execute api zero p schedule.order memory) (p.output.address i)=
      target zero count cols (readInput api p memory) (fun j=>api.read memory (p.output.address j)) i := by
  rw [execute_static]
  have h:=congrFun (static_readback api zero p (readInput api p memory) schedule.order memory) i
  rw [scheduled_words zero schedule positive,word_run_source] at h
  exact h.trans (logical_readback zero count cols positive _ _ i bound)

#axiomsClean tile_copy_same_source
#axiomsClean tile_order_independent
#axiomsClean scheduled_words
#axiomsClean execute_schedule_readback
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
