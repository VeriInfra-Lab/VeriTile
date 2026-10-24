import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalBackward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.FoldFacts
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole
namespace VeriTile.Bench.Repairs.LayerNormLiger.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def Reads (p : Backward.Parameters) (r : RegionName) :=
  r=p.x ∨ r=p.weight ∨ r=p.dy ∨ r=p.mean ∨ r=p.rstd
def InputEq (p : Backward.Parameters) (a b : Backward.Acc ℝ p) :=
  ∀ r, Reads p r → a.state.mem r=b.state.mem r
def FreshDX (p : Backward.Parameters) := ∀ r, Reads p r → r≠p.dx

theorem update_frame {α : Type} [Inhabited α] (M : Algebra α)
    (R : VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.SumPrimitive α)
    (p : Backward.Parameters) (a : Backward.Acc α p) (row : Nat) (r : RegionName) (o : Nat)
    (h : r≠p.dx ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o≠row*p.dxStride+i.1.val) :
    (Backward.update M R p row a).state.mem r o = a.state.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h|h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)

theorem update_inputs (rest : Algebra ℝ) (p : Backward.Parameters)
    (a ref : Backward.Acc ℝ p) (fresh : FreshDX p) (h : InputEq p a ref) (row : Nat) :
    InputEq p (Backward.update (algebra rest) sumPrimitive p row a) ref := by
  intro r hr
  funext o
  rw [update_frame _ _ _ _ _ _ _ (Or.inl (fresh r hr)), h r hr]

theorem input_values (p : Backward.Parameters) (a ref : Backward.Acc ℝ p) (h : InputEq p a ref) (row : Nat) :
    bx p a row=bx p ref row ∧ bw p a=bw p ref ∧ bdy p a row=bdy p ref row ∧
    Backward.mean p a row=Backward.mean p ref row ∧ Backward.rstd p a row=Backward.rstd p ref row := by
  have hx := h p.x (by simp [Reads])
  have hw := h p.weight (by simp [Reads])
  have hy := h p.dy (by simp [Reads])
  have hm := h p.mean (by simp [Reads])
  have hr := h p.rstd (by simp [Reads])
  refine ⟨?_,?_,?_,?_,?_⟩
  · funext i; simp [bx,hx]
  · funext i; simp [bw,hw]
  · funext i; simp [bdy,hy]
  · simp [Backward.mean,hm]
  · simp [Backward.rstd,hr]

theorem loop_dw (rest : Algebra ℝ) (p : Backward.Parameters) (a ref : Backward.Acc ℝ p)
    (fresh : FreshDX p) (h : InputEq p a ref) (cur stop : Nat)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (foldRange (Backward.update (algebra rest) sumPrimitive p) cur stop 1 a).dw i =
      a.dw i + ∑ row ∈ Finset.Ico cur stop,
        bdy p ref row i.1.val*xhat (bx p ref row) (Backward.mean p ref row) (Backward.rstd p ref row) i.1.val := by
  rw [Folds.project _ (fun row z => z + bdy p ref row i.1.val*
    xhat (bx p ref row) (Backward.mean p ref row) (Backward.rstd p ref row) i.1.val)
    (InputEq p · ref) (fun a => a.dw i) (fun n a ha => update_inputs rest p a ref fresh ha n) ?_ cur stop a h]
  · exact Folds.add_sum _ _ _ _
  · intro n a ha
    dsimp only
    rw [backward_dw_step rest p a n i hi]
    obtain ⟨hx,hw,hy,hm,hr⟩ := input_values p a ref ha n
    rw [hx,hy,hm,hr]

theorem loop_db (rest : Algebra ℝ) (p : Backward.Parameters) (a ref : Backward.Acc ℝ p)
    (fresh : FreshDX p) (h : InputEq p a ref) (cur stop : Nat)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (foldRange (Backward.update (algebra rest) sumPrimitive p) cur stop 1 a).db i =
      a.db i + ∑ row ∈ Finset.Ico cur stop, bdy p ref row i.1.val := by
  rw [Folds.project _ (fun row z => z + bdy p ref row i.1.val)
    (InputEq p · ref) (fun a => a.db i) (fun n a ha => update_inputs rest p a ref fresh ha n) ?_ cur stop a h]
  · exact Folds.add_sum _ _ _ _
  · intro n a ha
    dsimp only
    rw [backward_db_step rest p a n i hi, (input_values p a ref ha n).2.2.1]

theorem update_dx_cell (rest : Algebra ℝ) (p : Backward.Parameters) (a ref : Backward.Acc ℝ p)
    (h : InputEq p a ref) (hb : p.columns ≤ p.block) (hs : p.columns ≤ p.dxStride)
    (n row : Nat) (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (Backward.update (algebra rest) sumPrimitive p n a).state.mem p.dx (row*p.dxStride+i.1.val) =
      if n=row then .mk .fp32 (dx p.columns (bx p ref row) (bw p ref) (bdy p ref row)
        (Backward.mean p ref row) (Backward.rstd p ref row) i.1.val)
      else a.state.mem p.dx (row*p.dxStride+i.1.val) := by
  by_cases hn : n=row
  · subst n
    simp only [if_true]
    have hw : (Backward.update (algebra rest) sumPrimitive p row a).state.mem p.dx (row*p.dxStride+i.1.val) =
        Backward.stored (algebra rest) (Backward.dxValue (algebra rest) sumPrimitive p a row i) := by
      apply State.masked_scatter_readback
      · exact hi
      · intro x y hxy
        have hval : x.1.val=y.1.val := Nat.add_left_cancel hxy
        rcases x with ⟨x,⟨⟩⟩
        rcases y with ⟨y,⟨⟩⟩
        simpa [Fin.ext_iff] using hval
    rw [hw,backward_dx rest p a row hb i hi]
    obtain ⟨hx,hw,hy,hm,hr⟩ := input_values p a ref h row
    rw [hx,hw,hy,hm,hr]
    rfl
  · simp only [hn,if_false]
    apply update_frame
    right
    intro j hj he
    have hn' := (Launch.row_major_injective (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs) he).1
    exact hn hn'.symm

theorem loop_dx (rest : Algebra ℝ) (p : Backward.Parameters) (a ref : Backward.Acc ℝ p)
    (fresh : FreshDX p) (h : InputEq p a ref) (hb : p.columns ≤ p.block) (hs : p.columns ≤ p.dxStride)
    (cur stop row : Nat) (lo : cur ≤ row) (hr : row < stop)
    (i : TileIndex [p.block]) (hi : i.1.val < p.columns) :
    (foldRange (Backward.update (algebra rest) sumPrimitive p) cur stop 1 a).state.mem p.dx (row*p.dxStride+i.1.val) =
      .mk .fp32 (dx p.columns (bx p ref row) (bw p ref) (bdy p ref row)
        (Backward.mean p ref row) (Backward.rstd p ref row) i.1.val) := by
  rw [Folds.project _ (fun n z => if n=row then .mk .fp32
      (dx p.columns (bx p ref row) (bw p ref) (bdy p ref row)
        (Backward.mean p ref row) (Backward.rstd p ref row) i.1.val) else z)
    (InputEq p · ref) (fun a => a.state.mem p.dx (row*p.dxStride+i.1.val))
    (fun n a ha => update_inputs rest p a ref fresh ha n)
    (fun n a ha => update_dx_cell rest p a ref ha hb hs n row i hi) cur stop a h]
  exact Folds.set_inside row cur stop _ _ lo hr

theorem loop_frame {α : Type} [Inhabited α] (M : Algebra α)
    (R : VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.SumPrimitive α)
    (p : Backward.Parameters) (a : Backward.Acc α p) (cur stop : Nat) (r : RegionName) (o : Nat)
    (h : r≠p.dx ∨ ∀ row, cur ≤ row → row < stop →
      ∀ i : TileIndex [p.block], i.1.val < p.columns → o≠row*p.dxStride+i.1.val) :
    (foldRange (Backward.update M R p) cur stop 1 a).state.mem r o = a.state.mem r o := by
  by_cases hc : cur < stop
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hc,if_false,if_true,dite_false,dite_true]
    rw [loop_frame M R p (Backward.update M R p cur a) (cur+1) stop r o ?_]
    · exact update_frame M R p a cur r o (h.imp_right (fun hh => hh cur (Nat.le_refl _) hc))
    · exact h.imp_right (fun hh row hlo hhi => hh row (by omega) hhi)
  · simp [foldRange,hc]
termination_by stop-cur
decreasing_by omega

#axiomsClean update_frame
#axiomsClean update_inputs
#axiomsClean loop_dw
#axiomsClean loop_db
#axiomsClean loop_dx
#axiomsClean loop_frame
end
end VeriTile.Bench.Repairs.LayerNormLiger.Logical
