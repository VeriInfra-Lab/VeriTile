import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Frame
namespace VeriTile.Bench.Repairs.LayerNormOps
open VeriTile.Triton FP.Structural
noncomputable section
namespace Forward

theorem scatter_read {α : Type} (p : ForwardParameters) (row : Nat) (out : RegionName)
    (stride : Nat) (v : TileIndex [p.block] → Cell α) (s : State α)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (scatter p row out stride v s).mem out (row*stride+i.1.val) = v i := by
  unfold scatter
  apply State.masked_scatter_readback
  · exact hi
  · intro a b hab
    have h : a.1.val=b.1.val := Nat.add_left_cancel hab
    rcases a with ⟨a,⟨⟩⟩; rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using h

theorem reference_y {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (reference M p s).mem p.y (s.pids 0*p.yStride+i.1.val) = stored M (yValue M p s i) :=
  scatter_read p _ _ _ _ _ i hi

theorem reference_rstd {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (hy : p.rstd ≠ p.y) :
    (reference M p s).mem p.rstd (s.pids 0) = stored M (rstdValue M p s) := by
  unfold reference
  rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl hy)]
  simp [statisticsState,State.write]

theorem reference_mean {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (hr : p.rms = Bool.false) (hy : p.mean ≠ p.y) (hs : p.mean ≠ p.rstd) :
    (reference M p s).mem p.mean (s.pids 0) = stored M (meanValue M p s) := by
  unfold reference
  rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl hy)]
  simp [statisticsState,State.write,hr,hs]

theorem statistics_frame {α : Type} [Inhabited α] (M : Algebra α) (p : ForwardParameters)
    (s : State α) (r : RegionName) (o : Nat)
    (hs : r ≠ p.residualOut) (hm : r ≠ p.mean) (hr : r ≠ p.rstd) :
    (statisticsState M p s).mem r o = s.mem r o := by
  have hres : (residualState M p s).mem r o=s.mem r o := by
    cases hp : p.storeResidual
    · simp [residualState,hp]
    · simpa only [residualState,hp,if_true] using
        (scatter_frame p (s.pids 0) p.residualOut p.residualOutStride
          (fun i => stored M (z M p s i)) s r o (Or.inl hs))
  cases hp : p.rms <;> simp [statisticsState,hp,State.write,hm,hr,hres]
end Forward
namespace Backward

theorem scatter_read {α : Type} (p : BackwardParameters) (row : Nat) (out : RegionName)
    (stride : Nat) (v : TileIndex [p.block] → Cell α) (s : State α)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (scatter p row out stride v s).mem out (row*stride+i.1.val) = v i := by
  unfold scatter
  apply State.masked_scatter_readback
  · exact hi
  · intro a b hab
    have h : a.1.val=b.1.val := Nat.add_left_cancel hab
    rcases a with ⟨a,⟨⟩⟩; rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using h

theorem update_dx {α : Type} [Inhabited α] (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (update M p row a).state.mem p.dx (row*p.dxStride+i.1.val) = stored M (dxValue M p a row i) :=
  scatter_read p _ _ _ _ _ i hi

theorem finish_dw {α : Type} (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (pid : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns)
    (h : p.dw ≠ p.db) :
    (finish M p pid a).mem p.dw (pid*p.columns+i.1.val) = stored M (a.dw i) := by
  unfold finish
  cases hb : p.hasBias
  · simp only [hb,Bool.false_eq_true,if_false]
    exact scatter_read p _ _ _ _ _ i hi
  · simp only [hb,if_true]
    rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl h)]
    exact scatter_read p _ _ _ _ _ i hi

theorem finish_db {α : Type} (M : Algebra α) (p : BackwardParameters)
    (a : Acc α p) (pid : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns)
    (h : p.hasBias = Bool.true) :
    (finish M p pid a).mem p.db (pid*p.columns+i.1.val) = stored M (a.db i) := by
  simp only [finish,h,if_true]
  exact scatter_read p _ _ _ _ _ i hi
end Backward
#axiomsClean Forward.reference_y
#axiomsClean Forward.reference_mean
#axiomsClean Forward.reference_rstd
#axiomsClean Backward.update_dx
#axiomsClean Backward.finish_dw
#axiomsClean Backward.finish_db
end
end VeriTile.Bench.Repairs.LayerNormOps
