import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedRowPost

/-! Ordinary disjoint-row launch of the actual typed source. Every row's
result is produced by `FP.Structural.exec`; merging reads those result cells,
not a mathematical oracle. Empty grids preserve the entire initial memory. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def atRow {α : Type} (s : State α) (M row : Nat) : State α :=
  { s with pids := fun ax => if ax=0 then row else 0
           numPids := fun ax => if ax=0 then M else 1 }

def RowValues {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block row : Nat) (s t : State α) : Prop :=
  let v := combinedValues A cmp f s r stride M N p row block
  InputPost A cmp f r stride M N p block row s t ∧
  (∀ j : TileIndex [block], j.1.val<N →
    t.mem r.y (row*stride.y+j.1.val)=Cell.mk .real
      (affineValue A f.bias (normalizedValues A f.rms N eps v j)
        (raw A s r.weight 0 N j.1.val) (raw A s r.bias 0 N j.1.val))) ∧
  (f.w1=Bool.true → ∀ j : TileIndex [block], j.1.val<N →
    t.mem r.y1 (row*stride.y1+j.1.val)=Cell.mk .real
      (affineValue A f.b1 (normalizedValues A f.rms N eps v j)
        (raw A s r.weight1 0 N j.1.val) (raw A s r.bias1 0 N j.1.val))) ∧
  t.mem r.rstd row=Cell.mk .real (cast A (rstdValue A f.rms N eps v)) ∧
  (f.rms=Bool.false → t.mem r.mean row=Cell.mk .real (cast A (meanValue A N v)))

structure GridRun {α : Type} [Inhabited α] (A : Algebra α) (f : Flags) (r : Regions)
    (stride : Strides) (M N : Nat) (eps p : ℝ) (block : Nat)
    (hb : Random.ArangeSizeDomain block) (initial final : State α) where
  rows : Fin M → State α
  executed : ∀ i, FP.Structural.exec (engine A) (repaired f r stride M N eps p block hb)
    (atRow initial M i.val)=some (rows i)
  framed : ∀ i, Frame f r stride M N i.val (atRow initial M i.val) (rows i)
  disjoint : ∀ (i j : Fin M) region offset,
    Contract.written f r stride M N i.val region offset →
    Contract.written f r stride M N j.val region offset → i=j
  readback : ∀ (i : Fin M) region offset,
    Contract.written f r stride M N i.val region offset →
    final.mem region offset=(rows i).mem region offset
  frame : ∀ region offset,
    (∀ i : Fin M, ¬ Contract.written f r stride M N i.val region offset) →
    final.mem region offset=initial.mem region offset

def GridPost {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (s t : State α) : Prop :=
  (∀ i : Fin M, RowValues A cmp f r stride M N eps p block i.val s t) ∧
  (∀ region offset, (∀ i : Fin M, ¬ Contract.written f r stride M N i.val region offset) →
    t.mem region offset=s.mem region offset)

theorem RowValues.transfer {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block row : Nat) (s u t : State α)
    (hu : RowValues A cmp f r stride M N eps p block row s u)
    (ht : ∀ region offset, Contract.written f r stride M N row region offset →
      t.mem region offset=u.mem region offset) :
    RowValues A cmp f r stride M N eps p block row s t := by
  rcases hu with ⟨⟨hres,hm1,hm2⟩,hy,hy1,hstd,hm⟩
  refine ⟨⟨?_,?_,?_⟩,?_,?_,?_,?_⟩
  · intro hf j hj
    rw [ht _ _ (by simp [Contract.written,hf,Contract.inRow_of_col hj])]
    exact hres hf j hj
  · intro hd hmask j hj
    rw [ht _ _ (by simp [Contract.written,hd,hmask,Contract.inRow_of_col hj])]
    exact hm1 hd hmask j hj
  · intro hd hmask hx1 j hj
    rw [ht _ _ (by simp [Contract.written,hd,hmask,hx1,Contract.inRow_of_col hj])]
    exact hm2 hd hmask hx1 j hj
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

theorem source_grid_post {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hgeo : WholeGrid.Geometry r stride N) :
    ∃ t, Nonempty (GridRun A f r stride M N eps p block hb s t) ∧
      GridPost A cmp f r stride M N eps p block s t := by
  have hrows : ∀ i : Fin M, ∃ t,
      FP.Structural.exec (engine A) (repaired f r stride M N eps p block hb) (atRow s M i.val)=some t ∧
      RowPost A cmp f r stride M N eps p block i.val (atRow s M i.val) t := by
    intro i
    simpa [atRow] using source_row_post A cmp f r stride M N eps p block hb (atRow s M i.val) hsep hgeo (by simpa [atRow] using i.isLt)
  choose rows hexec hpost using hrows
  let merged : State α :=
    {s with mem := fun region offset =>
      if h : (∃ i : Fin M, Contract.written f r stride M N i.val region offset) then
        ((rows h.choose).mem region offset)
      else s.mem region offset}
  have hdis : ∀ (i j : Fin M) region offset,
      Contract.written f r stride M N i.val region offset →
      Contract.written f r stride M N j.val region offset → i=j := by
    intro i j region offset hi hj
    exact Fin.ext (WholeGrid.written_row_unique f r stride M N i.val j.val hgeo i.isLt j.isLt region offset hi hj)
  have hread : ∀ (i : Fin M) region offset,
      Contract.written f r stride M N i.val region offset →
      merged.mem region offset=(rows i).mem region offset := by
    intro i region offset hi
    have hex : ∃ i : Fin M, Contract.written f r stride M N i.val region offset := ⟨i,hi⟩
    change (if h : (∃ i : Fin M, Contract.written f r stride M N i.val region offset) then
      (rows h.choose).mem region offset else s.mem region offset)=_
    rw [dif_pos hex,hdis hex.choose i region offset hex.choose_spec hi]
  have hframe : ∀ region offset,
      (∀ i : Fin M, ¬ Contract.written f r stride M N i.val region offset) →
      merged.mem region offset=s.mem region offset := by
    intro region offset h
    dsimp only [merged]
    rw [dif_neg (by simpa only [not_exists] using h)]
  refine ⟨merged,⟨⟨rows,hexec,fun i => (hpost i).1,hdis,hread,hframe⟩⟩,?_,hframe⟩
  intro i
  have hv : RowValues A cmp f r stride M N eps p block i.val s (rows i) := (hpost i).2
  exact hv.transfer A cmp f r stride M N eps p block i.val s (rows i) merged (hread i)

theorem zero_rows_frame {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (N : Nat) (eps p : ℝ)
    (block : Nat) (s t : State α) (h : GridPost A cmp f r stride 0 N eps p block s t) :
    t.mem=s.mem := by
  funext region offset
  exact h.2 region offset (fun i => Fin.elim0 i)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
