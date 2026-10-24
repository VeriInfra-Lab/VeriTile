import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeRowPost

/-! Ordinary disjoint-row execution of the explicit native schedule. The grid
merges the actual cells stored by every row, never an oracle-supplied value. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

def executeGrid {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (hblock : 0<block) (s : State α) : State α :=
  {s with mem := fun region offset =>
    if h : ∃ i : Fin M, Contract.written f r stride M N i.val region offset then
      (execute a red plan plan1 f r stride M N h.choose.val eps p hblock s).mem region offset
    else s.mem region offset}

def GridPost {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (s t : State α) : Prop :=
  (∀ i : Fin M, RowValues a red plan plan1 f r stride M N i.val eps p s t) ∧
  (∀ region offset, (∀ i : Fin M, ¬Contract.written f r stride M N i.val region offset) →
    t.mem region offset=s.mem region offset)

structure GridRun {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (hblock : 0<block) (s t : State α) where
  rows : Fin M → State α
  executed : ∀ i, rows i=execute a red plan plan1 f r stride M N i.val eps p hblock s
  framed : ∀ i, TypedSource.Frame f r stride M N i.val s (rows i)
  disjoint : ∀ (i j : Fin M) region offset,
    Contract.written f r stride M N i.val region offset →
    Contract.written f r stride M N j.val region offset → i=j
  readback : ∀ (i : Fin M) region offset,
    Contract.written f r stride M N i.val region offset → t.mem region offset=(rows i).mem region offset
  frame : ∀ region offset, (∀ i : Fin M, ¬Contract.written f r stride M N i.val region offset) →
    t.mem region offset=s.mem region offset

theorem RowValues.transfer {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N row : Nat)
    (eps p : α) (s u t : State α) (hu : RowValues a red plan plan1 f r stride M N row eps p s u)
    (ht : ∀ region offset, Contract.written f r stride M N row region offset → t.mem region offset=u.mem region offset) :
    RowValues a red plan plan1 f r stride M N row eps p s t := by
  rcases hu with ⟨hres,hm1,hm2,hy,hy1,hstd,hm⟩
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · intro hf j hj
    rw [ht _ _ (by simp [Contract.written,hf,Contract.inRow_of_col hj])]
    exact hres hf j hj
  · intro hd hmask j hj
    rw [ht _ _ (by simp [Contract.written,hd,hmask,Contract.inRow_of_col hj])]
    exact hm1 hd hmask j hj
  · intro hd hmask hx j hj
    rw [ht _ _ (by simp [Contract.written,hd,hmask,hx,Contract.inRow_of_col hj])]
    exact hm2 hd hmask hx j hj
  · intro j hj
    rw [ht _ _ (by simp [Contract.written,Contract.inRow_of_col hj])]
    exact hy j hj
  · intro hf j hj
    rw [ht _ _ (by simp [Contract.written,hf,Contract.inRow_of_col hj])]
    exact hy1 hf j hj
  · rw [ht _ _ (by simp [Contract.written])]
    exact hstd
  · intro hf
    rw [ht _ _ (by simp [Contract.written,hf])]
    exact hm hf

theorem execute_grid_post {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : α) (hblock : 0<block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hgeo : NativeGeometry.Geometry f r stride M N) :
    Nonempty (GridRun a red plan plan1 f r stride M N eps p hblock s
      (executeGrid a red plan plan1 f r stride M N eps p hblock s)) ∧
    GridPost a red plan plan1 f r stride M N eps p s
      (executeGrid a red plan plan1 f r stride M N eps p hblock s) := by
  let rows := fun i : Fin M => execute a red plan plan1 f r stride M N i.val eps p hblock s
  let final := executeGrid a red plan plan1 f r stride M N eps p hblock s
  have hpost := fun i : Fin M => execute_row_post a red plan plan1 f r stride M N i.val eps p hblock s hsep hgeo i.isLt
  have hdis : ∀ (i j : Fin M) region offset,
      Contract.written f r stride M N i.val region offset →
      Contract.written f r stride M N j.val region offset → i=j := by
    intro i j region offset hi hj
    exact Fin.ext (NativeGeometry.written_row_unique f r stride M N i.val j.val hgeo i.isLt j.isLt region offset hi hj)
  have hread : ∀ (i : Fin M) region offset, Contract.written f r stride M N i.val region offset →
      final.mem region offset=(rows i).mem region offset := by
    intro i region offset hi
    have hex : ∃ i : Fin M, Contract.written f r stride M N i.val region offset := ⟨i,hi⟩
    change (if h : ∃ i : Fin M, Contract.written f r stride M N i.val region offset then
      (rows h.choose).mem region offset else s.mem region offset)=_
    rw [dif_pos hex,hdis hex.choose i region offset hex.choose_spec hi]
  have hframe : ∀ region offset, (∀ i : Fin M, ¬Contract.written f r stride M N i.val region offset) →
      final.mem region offset=s.mem region offset := by
    intro region offset h
    dsimp only [final,executeGrid]
    rw [dif_neg (by simpa only [not_exists] using h)]
  refine ⟨⟨⟨rows,fun _ => rfl,fun i => (hpost i).1,hdis,hread,hframe⟩⟩,?_,hframe⟩
  intro i
  exact RowValues.transfer a red plan plan1 f r stride M N i.val eps p s (rows i) final (hpost i).2 (hread i)

theorem empty_grid_frame {α : Type} [Inhabited α] (a : Ops α) (red : Nat → Whole.Reductions block)
    (plan plan1 : AffinePlan) (f : Flags) (r : Regions) (stride : Strides) (N : Nat)
    (eps p : α) (hblock : 0<block) (s : State α) :
    executeGrid a red plan plan1 f r stride 0 N eps p hblock s=s := by
  have hnone : ∀ region offset, ¬∃ i : Fin 0, Contract.written f r stride 0 N i.val region offset := by simp
  simp [executeGrid,hnone]

#axiomsClean execute_grid_post
#axiomsClean empty_grid_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeSchedule
