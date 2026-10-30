import bench.tritonbench_g.context_attn_mistral.Proofs.OutputView

namespace VeriTile.Bench.Repairs.ContextAttnMistral.OutputView
noncomputable section

/-- A finite snapshot of slice descriptors is fixed before any output write.
Each copy reads the same private scratch. This is the wrapper's ordered loop. -/
def View.copySlices {C V S : Type} (view : View C V) (active : S → C → Prop)
    (value : C → V) : List S → (Nat → V) → (Nat → V)
  | [], memory => memory
  | span :: rest, memory => view.copySlices active value rest (view.copy (active span) memory value)

def SliceActive {C S : Type} (active : S → C → Prop) (spans : List S) (i : C) : Prop :=
  ∃ span ∈ spans, active span i

theorem copySlices_frame {C V S : Type} (view : View C V) (active : S → C → Prop)
    (value : C → V) (spans : List S) (memory : Nat → V) (a : Nat)
    (ha : ¬ ∃ i, SliceActive active spans i ∧ view.addr i = a) :
    view.copySlices active value spans memory a = memory a := by
  induction spans generalizing memory with
  | nil => rfl
  | cons span rest ih =>
    have hrest : ¬ ∃ i, SliceActive active rest i ∧ view.addr i = a := by
      rintro ⟨i,⟨s,hs,hi⟩,he⟩
      exact ha ⟨i,⟨s,List.mem_cons_of_mem span hs,hi⟩,he⟩
    have hspan : ¬ ∃ i, active span i ∧ view.addr i = a := by
      rintro ⟨i,hi,he⟩
      exact ha ⟨i,⟨span,List.mem_cons_self,hi⟩,he⟩
    rw [View.copySlices,ih _ hrest,copy_frame _ _ _ _ _ hspan]

theorem copySlices_at {C V S : Type} (view : View C V) (active : S → C → Prop)
    (value : C → V) (spans : List S) (memory : Nat → V)
    (hw : view.Writable (SliceActive active spans)) (i : C)
    (hi : SliceActive active spans i) :
    view.read (view.copySlices active value spans memory) i = value i := by
  classical
  induction spans generalizing memory with
  | nil => obtain ⟨s,hs,_⟩ := hi; simp at hs
  | cons span rest ih =>
    have liftRest : ∀ j, SliceActive active rest j → SliceActive active (span :: rest) j := by
      rintro j ⟨s,hs,hj⟩
      exact ⟨s,List.mem_cons_of_mem span hs,hj⟩
    by_cases hr : SliceActive active rest i
    · exact ih _ (fun j k hj hk he => hw j k (liftRest j hj) (liftRest k hk) he) hr
    · have hs : active span i := by
        obtain ⟨s,hs,hsi⟩ := hi
        rcases List.mem_cons.mp hs with he | he
        · simpa [he] using hsi
        · exact False.elim (hr ⟨s,he,hsi⟩)
      have hf : ¬ ∃ j, SliceActive active rest j ∧ view.addr j = view.addr i := by
        rintro ⟨j,hj,he⟩
        have hji := hw j i (liftRest j hj) hi he
        exact hr (hji ▸ hj)
      change view.decode (view.copySlices active value rest
        (view.copy (active span) memory value) (view.addr i)) = value i
      rw [copySlices_frame _ _ _ _ _ _ hf]
      exact copy_at view (active span)
        (fun j k hj hk he => hw j k ⟨span,List.mem_cons_self,hj⟩ ⟨span,List.mem_cons_self,hk⟩ he)
        memory value i hs

/-- The ordered wrapper loop implements the active-union copy contract used
by the independent Real proof. This equality is on every physical address,
including inactive coordinates that alias active output locations. -/
theorem copySlices_eq_copy {C V S : Type} (view : View C V) (active : S → C → Prop)
    (value : C → V) (spans : List S) (memory : Nat → V)
    (hw : view.Writable (SliceActive active spans)) :
    view.copySlices active value spans memory = view.copy (SliceActive active spans) memory value := by
  classical
  funext a
  by_cases ha : ∃ i, SliceActive active spans i ∧ view.addr i = a
  · obtain ⟨i,hi,he⟩ := ha
    have hleft := congrArg view.encode (copySlices_at view active value spans memory hw i hi)
    have hright := congrArg view.encode (copy_at view (SliceActive active spans) hw memory value i hi)
    simp only [View.read,view.encode_decode,he] at hleft hright
    exact hleft.trans hright.symm
  · rw [copySlices_frame _ _ _ _ _ _ ha,copy_frame _ _ _ _ _ ha]

end
end VeriTile.Bench.Repairs.ContextAttnMistral.OutputView
