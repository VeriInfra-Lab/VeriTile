import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalBackward
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Scope
namespace VeriTile.Bench.Repairs.LayerNormOps.Logical
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

theorem forward_y (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns≤p.block) (i : TileIndex [p.block]) (hi : i.1.val<p.columns)
    (hw : p.weight≠p.residualOut ∧ p.weight≠p.mean ∧ p.weight≠p.rstd)
    (hbi : p.bias≠p.residualOut ∧ p.bias≠p.mean ∧ p.bias≠p.rstd) :
    Forward.yValue (algebra rest) p s i = y p.columns p.rms p.hasBias p.epsilon (input p s)
      (fun j => (s.mem p.weight j).read .fp32) (fun j => (s.mem p.bias j).read .fp32) i.1.val := by
  unfold Forward.yValue Forward.unspecified
  dsimp only
  rw [Forward.statistics_frame _ _ _ _ _ hw.1 hw.2.1 hw.2.2,
    Forward.statistics_frame _ _ _ _ _ hbi.1 hbi.2.1 hbi.2.2]
  rw [z_value,mean_value rest p s hb,rstd_value rest p s hb]
  cases hr : p.rms <;> cases hbb : p.hasBias <;>
    simp [hi,hr,hbb,y,normalized,mean,Forward.cast,Forward.unhintedBinary,
      algebra,LayerNormLiger.Logical.algebra]

theorem backward_dx (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val < p.columns → a.weight i=w i.1.val)
    (hb : p.columns ≤ p.block) (i : TileIndex [p.block]) (hi : i.1.val < p.columns)
    (hfresh : p.dres≠p.y) :
    Backward.dxValue (algebra rest) p a row i =
      dx p.columns p.rms p.hasDres (bx p a row) w (bdy p a row)
        (fun j => (a.state.mem p.dres (row*p.dresStride+j)).read .fp32)
        (bmu p a row) (binv p a row) i.1.val := by
  have hf : (Backward.recomputed (algebra rest) p a row).mem p.dres (row*p.dresStride+i.1.val) =
      a.state.mem p.dres (row*p.dresStride+i.1.val) := by
    cases hy : p.recompute
    · simp [Backward.recomputed,hy]
    · simp only [Backward.recomputed,hy,if_true]
      exact Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hfresh)
  unfold Backward.dxValue
  rw [backward_dxBase rest p a row w hw hb i hi]
  cases hd : p.hasDres <;>
    simp [hd,Backward.loaded,hi,hf,Forward.cast,Forward.binary,
      algebra,LayerNormLiger.Logical.algebra,dx]
  all_goals exact congrArg (Cell.read .fp32) hf

/-- The source execution produces both statistics and Y; no saved-statistic
correctness assumption enters this theorem. -/
theorem forward_produced {rest : Algebra ℝ} (p : ForwardParameters) (s : State ℝ)
    (hb : p.columns≤p.block) (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y)
    (hw : p.weight≠p.residualOut ∧ p.weight≠p.mean ∧ p.weight≠p.rstd)
    (hbi : p.bias≠p.residualOut ∧ p.bias≠p.mean ∧ p.bias≠p.rstd) :
    ∃ t, FP.Structural.exec (algebra rest) (forwardKernel p) s=some t ∧
      (t.mem p.rstd (s.pids 0)).read .fp32=rstd p.columns p.rms p.epsilon (input p s) ∧
      (p.rms=Bool.false → (t.mem p.mean (s.pids 0)).read .fp32=LayerNormLiger.Logical.sum p.columns (input p s)/p.columns) ∧
      ∀ i : TileIndex [p.block], i.1.val<p.columns →
        (t.mem p.y (s.pids 0*p.yStride+i.1.val)).read .fp32=
          y p.columns p.rms p.hasBias p.epsilon (input p s)
            (fun j => (s.mem p.weight j).read .fp32) (fun j => (s.mem p.bias j).read .fp32) i.1.val := by
  obtain ⟨t,ht,hm⟩ := Forward.complete_run (algebra rest) p s
  refine ⟨t,ht,?_,?_,?_⟩
  · rw [hm,Forward.reference_rstd _ _ _ hry]
    simpa [Forward.stored,algebra,LayerNormLiger.Logical.algebra,Cell.read] using rstd_value rest p s hb
  · intro hr
    rw [hm,Forward.reference_mean _ _ _ hr hmy hmr]
    simpa [Forward.stored,algebra,LayerNormLiger.Logical.algebra,Cell.read] using mean_value rest p s hb
  · intro i hi
    rw [hm,Forward.reference_y _ _ _ i hi]
    simpa [Forward.stored,algebra,LayerNormLiger.Logical.algebra,Cell.read] using forward_y rest p s hb i hi hw hbi

#axiomsClean forward_y
#axiomsClean backward_dx
#axiomsClean forward_produced
end
end VeriTile.Bench.Repairs.LayerNormOps.Logical
