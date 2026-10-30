import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelFrame

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelReadback
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

structure Separated (p : BackwardParameters) : Prop where
  dx_y : p.dx≠p.y
  dx_dr : p.dx≠p.dresIn
  y_dr : p.y≠p.dresIn
  dx_dw : p.dx≠p.dw
  dx_db : p.dx≠p.db
  y_dw : p.y≠p.dw
  y_db : p.y≠p.db
  dr_dw : p.dresIn≠p.dw
  dr_db : p.dresIn≠p.db
  dw_db : p.dw≠p.db

structure Admissible (p : BackwardParameters) : Prop where
  fresh : ∀ r∈KernelInvariant.reads p,Backward.RegionOutside p r
  separated : Separated p
  dxStride : p.columns≤p.dxStride
  yStride : p.recompute=Bool.true → p.columns≤p.yStride
  drStride : p.storeDres=Bool.true → p.columns≤p.dresInStride

theorem dres_y (p : BackwardParameters) (h : Admissible p) :
    p.recompute=Bool.true → p.dres≠p.y :=
  (h.fresh p.dres (by simp [KernelInvariant.reads])).2.2.2.1

theorem update_y (A : Algebra α) (p : BackwardParameters) (s : State α)
    (a : Backward.Acc α p) (ha : KernelInvariant.Holds A p s a) (h : Separated p)
    (row : Nat) (hy : p.recompute=Bool.true)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.update A p row a).state.mem p.y (row*p.yStride+i.1.val)=
      Forward.stored A (KernelReference.rowY A p s row i) := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.dx_y.symm)]
  have hv : (Backward.recomputed A p a row).mem p.y (row*p.yStride+i.1.val)=
      Forward.stored A (KernelReference.rowY A p s row i) := by
    simp only [Backward.recomputed,hy,if_true]
    rw [Backward.scatter_read _ _ _ _ _ _ i hi,KernelInvariant.y A p s a ha row i]
  cases hd : p.storeDres
  · simpa only [hd,Bool.false_eq_true,if_false] using hv
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.y_dr)]
    exact hv

theorem update_dr (A : Algebra α) (p : BackwardParameters) (s : State α)
    (a : Backward.Acc α p) (ha : KernelInvariant.Holds A p s a) (h : Admissible p)
    (row : Nat) (hd : p.storeDres=Bool.true)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.update A p row a).state.mem p.dresIn (row*p.dresInStride+i.1.val)=
      Forward.stored A (KernelReference.rowDX A p s row i) := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.separated.dx_dr.symm)]
  simp only [hd,if_true]
  rw [Backward.scatter_read _ _ _ _ _ _ i hi,KernelInvariant.dx A p s a ha row i (dres_y p h)]

theorem row_dx_other (p : BackwardParameters) (h : Admissible p) (row other : Nat)
    (hne : other≠row) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelFrame.rowOutside p other p.dx (row*p.dxStride+i.1.val) := by
  refine ⟨Or.inr ?_,fun _ => Or.inl h.separated.dx_y,fun _ => Or.inl h.separated.dx_dr⟩
  intro j hj
  exact KernelFrame.offset_ne _ _ _ _ _ (Nat.lt_of_lt_of_le hi h.dxStride)
    (Nat.lt_of_lt_of_le hj h.dxStride) hne

theorem row_y_other (p : BackwardParameters) (h : Admissible p) (row other : Nat)
    (hy : p.recompute=Bool.true) (hne : other≠row) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelFrame.rowOutside p other p.y (row*p.yStride+i.1.val) := by
  refine ⟨Or.inl h.separated.dx_y.symm,fun _ => Or.inr ?_,fun _ => Or.inl h.separated.y_dr⟩
  intro j hj
  exact KernelFrame.offset_ne _ _ _ _ _ (Nat.lt_of_lt_of_le hi (h.yStride hy))
    (Nat.lt_of_lt_of_le hj (h.yStride hy)) hne

theorem row_dr_other (p : BackwardParameters) (h : Admissible p) (row other : Nat)
    (hd : p.storeDres=Bool.true) (hne : other≠row) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelFrame.rowOutside p other p.dresIn (row*p.dresInStride+i.1.val) := by
  refine ⟨Or.inl h.separated.dx_dr.symm,fun _ => Or.inl h.separated.y_dr.symm,fun _ => Or.inr ?_⟩
  intro j hj
  exact KernelFrame.offset_ne _ _ _ _ _ (Nat.lt_of_lt_of_le hi (h.drStride hd))
    (Nat.lt_of_lt_of_le hj (h.drStride hd)) hne

theorem loop_inside (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (r : RegionName) (o row : Nat) (v : Cell α)
    (hhere : ∀ a,KernelInvariant.Holds A p s a → (Backward.update A p row a).state.mem r o=v)
    (hother : ∀ other,other≠row → KernelFrame.rowOutside p other r o)
    (lo : Backward.start p s≤row) (hi : row<Backward.stop p s) :
    (Backward.accumulated A p s).state.mem r o=v := by
  have hp : (Backward.accumulated A p s).state.mem r o=
      foldRange (fun n value => if n=row then v else value)
        (Backward.start p s) (Backward.stop p s) 1 (s.mem r o) := by
    apply Folds.project (Backward.update A p) _ (KernelInvariant.Holds A p s)
      (fun a => a.state.mem r o) (KernelInvariant.update A p s h.fresh) ?_ _ _ _
      (KernelInvariant.initial A p s)
    intro n a ha
    dsimp only
    by_cases hn : n=row
    · subst n
      rw [if_pos rfl,hhere a ha]
    · rw [if_neg hn,KernelFrame.update A p n a r o (hother n hn)]
  rw [hp]
  exact Folds.set_inside _ _ _ _ _ lo hi

theorem reference_dx (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (row : Nat) (lo : Backward.start p s≤row) (hr : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference A p s).mem p.dx (row*p.dxStride+i.1.val)=
      Forward.stored A (KernelReference.rowDX A p s row i) := by
  rw [Backward.reference,KernelFrame.finish A p _ _ _ _
    (Or.inl h.separated.dx_dw) (fun _ => Or.inl h.separated.dx_db)]
  apply loop_inside A p s h _ _ row _ ?_ (fun n hn => row_dx_other p h row n hn i hi) lo hr
  intro a ha
  rw [Backward.update_dx _ _ _ _ i hi,KernelInvariant.dx A p s a ha row i (dres_y p h)]

theorem reference_y (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (hy : p.recompute=Bool.true)
    (row : Nat) (lo : Backward.start p s≤row) (hr : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference A p s).mem p.y (row*p.yStride+i.1.val)=
      Forward.stored A (KernelReference.rowY A p s row i) := by
  rw [Backward.reference,KernelFrame.finish A p _ _ _ _
    (Or.inl h.separated.y_dw) (fun _ => Or.inl h.separated.y_db)]
  exact loop_inside A p s h _ _ row _ (fun a ha => update_y A p s a ha h.separated row hy i hi)
    (fun n hn => row_y_other p h row n hy hn i hi) lo hr

theorem reference_dr (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (hd : p.storeDres=Bool.true)
    (row : Nat) (lo : Backward.start p s≤row) (hr : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference A p s).mem p.dresIn (row*p.dresInStride+i.1.val)=
      Forward.stored A (KernelReference.rowDX A p s row i) := by
  rw [Backward.reference,KernelFrame.finish A p _ _ _ _
    (Or.inl h.separated.dr_dw) (fun _ => Or.inl h.separated.dr_db)]
  exact loop_inside A p s h _ _ row _ (fun a ha => update_dr A p s a ha h row hd i hi)
    (fun n hn => row_dr_other p h row n hd hn i hi) lo hr

theorem reference_dw (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference A p s).mem p.dw (s.pids 0*p.columns+i.1.val)=
      Forward.stored A ((KernelReference.partials A p s).1 i) := by
  rw [Backward.reference,Backward.finish_dw A p _ _ i hi h.separated.dw_db]
  exact congrArg (fun a => Forward.stored A (a.1 i)) (KernelInvariant.partials A p s h.fresh)

theorem reference_db (A : Algebra α) (p : BackwardParameters) (s : State α)
    (h : Admissible p) (hb : p.hasBias=Bool.true) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.reference A p s).mem p.db (s.pids 0*p.columns+i.1.val)=
      Forward.stored A ((KernelReference.partials A p s).2 i) := by
  rw [Backward.reference,Backward.finish_db A p _ _ i hi hb]
  exact congrArg (fun a => Forward.stored A (a.2 i)) (KernelInvariant.partials A p s h.fresh)

theorem forward_residual (A : Algebra α) (p : ForwardParameters) (s : State α)
    (hs : p.storeResidual=Bool.true) (hy : p.residualOut≠p.y)
    (hm : p.residualOut≠p.mean) (hr : p.residualOut≠p.rstd)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Forward.reference A p s).mem p.residualOut (s.pids 0*p.residualOutStride+i.1.val)=
      Forward.stored A (Forward.z A p s i) := by
  unfold Forward.reference
  rw [Forward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hy)]
  have hv : (Forward.residualState A p s).mem p.residualOut (s.pids 0*p.residualOutStride+i.1.val)=
      Forward.stored A (Forward.z A p s i) := by
    simp only [Forward.residualState,hs,if_true]
    exact Forward.scatter_read _ _ _ _ _ _ i hi
  cases he : p.rms <;> simpa [Forward.statisticsState,he,State.write,hm,hr] using hv

#axiomsClean reference_dx
#axiomsClean reference_y
#axiomsClean reference_dr
#axiomsClean reference_dw
#axiomsClean reference_db
#axiomsClean forward_residual
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelReadback
