import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathModel
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 1000000

 theorem forward_input (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (count row : Nat) (hx : Host.ForwardRegionOutside p p.x)
    (hr : Host.ForwardRegionOutside p p.residual) :
    Logical.input p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s count).mem row) =
      z p s row := by
  funext j
  simp only [Logical.input,z,Grid.launch,if_true]
  rw [Host.forward_grid_frame _ _ _ _ _ _ hx,Host.forward_grid_frame _ _ _ _ _ _ hr]

 theorem forward_y_grid (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (rows row : Nat) (hrow : row<rows) (hb : p.columns≤p.block) (hs : p.columns≤p.yStride)
    (hd : [p.y,p.mean,p.rstd,p.residualOut].Nodup)
    (hf : ∀ r ∈ [p.x,p.residual,p.weight,p.bias], Host.ForwardRegionOutside p r)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    ((Grid.reference (Forward.reference (Logical.algebra rest) p) s rows).mem
      p.y (row*p.yStride+i.1.val)).read .fp32 = y p s row i.1.val := by
  have hym : p.y≠p.mean := by simp_all
  have hyr : p.y≠p.rstd := by simp_all
  have hyz : p.y≠p.residualOut := by simp_all
  rw [Grid.readback _ s rows row hrow p.y (row*p.yStride+i.1.val) ?_]
  · trans Forward.yValue (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) i
    · simpa only [Grid.launch,if_true,Forward.stored,Logical.algebra,LayerNormLiger.Logical.algebra,Cell.read] using
        congrArg (Cell.read .fp32) (Forward.reference_y (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) i hi)
    · have hw := hf p.weight (by simp)
      have hbi := hf p.bias (by simp)
      rw [Logical.forward_y rest p _ hb i hi ⟨hw.2.1,hw.2.2.1,hw.2.2.2⟩
        ⟨hbi.2.1,hbi.2.2.1,hbi.2.2.2⟩,
        forward_input rest p s row row (hf p.x (by simp)) (hf p.residual (by simp))]
      unfold y weight bias
      congr 1 <;> funext j <;> simp only [Grid.launch]
      · rw [Host.forward_grid_frame _ _ _ _ _ _ hw]
      · rw [Host.forward_grid_frame _ _ _ _ _ _ hbi]
  · intro later hl hu mem
    apply Forward.reference_frame
    refine ⟨Or.inr ?_,fun _ => Or.inl hyz,fun _ => Or.inl hym,Or.inl hyr⟩
    intro j hj
    exact Grid.row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs)
      row later (by omega)

 theorem forward_rstd_grid (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (rows row : Nat) (hrow : row<rows) (hb : p.columns≤p.block)
    (hd : [p.y,p.mean,p.rstd,p.residualOut].Nodup)
    (hx : Host.ForwardRegionOutside p p.x) (hr : Host.ForwardRegionOutside p p.residual) :
    ((Grid.reference (Forward.reference (Logical.algebra rest) p) s rows).mem p.rstd row).read .fp32 = inv p s row := by
  have hry : p.rstd≠p.y := by simp_all [ne_comm]
  have hrm : p.rstd≠p.mean := by simp_all [ne_comm]
  have hrz : p.rstd≠p.residualOut := by simp_all
  rw [Grid.readback _ s rows row hrow p.rstd row ?_]
  · trans Forward.rstdValue (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row)
    · simpa only [Grid.launch,if_true,Forward.stored,Logical.algebra,LayerNormLiger.Logical.algebra,Cell.read] using
        congrArg (Cell.read .fp32) (Forward.reference_rstd (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) hry)
    · rw [Logical.rstd_value rest p _ hb,forward_input rest p s row row hx hr]
      rfl
  · intro later hl hu mem
    apply Forward.reference_frame
    exact ⟨Or.inl hry,fun _ => Or.inl hrz,fun _ => Or.inl hrm,Or.inr (by simp [Grid.launch]; omega)⟩

 theorem forward_mean_grid (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (rows row : Nat) (hrow : row<rows) (hb : p.columns≤p.block)
    (hd : [p.y,p.mean,p.rstd,p.residualOut].Nodup) (plain : p.rms=Bool.false)
    (hx : Host.ForwardRegionOutside p p.x) (hr : Host.ForwardRegionOutside p p.residual) :
    ((Grid.reference (Forward.reference (Logical.algebra rest) p) s rows).mem p.mean row).read .fp32 = mu p s row := by
  have hmy : p.mean≠p.y := by simp_all [ne_comm]
  have hmr : p.mean≠p.rstd := by simp_all
  have hmz : p.mean≠p.residualOut := by simp_all
  rw [Grid.readback _ s rows row hrow p.mean row ?_]
  · trans Forward.meanValue (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row)
    · simpa only [Grid.launch,if_true,Forward.stored,Logical.algebra,LayerNormLiger.Logical.algebra,Cell.read] using
        congrArg (Cell.read .fp32) (Forward.reference_mean (Logical.algebra rest) p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) plain hmy hmr)
    · rw [Logical.mean_value rest p _ hb,forward_input rest p s row row hx hr]
      simp [mu,Logical.mean,plain]
  · intro later hl hu mem
    apply Forward.reference_frame
    exact ⟨Or.inl hmy,fun _ => Or.inl hmz,fun _ => Or.inr (by simp [Grid.launch]; omega),Or.inl hmr⟩

 theorem reference_residual (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (hs : p.storeResidual=Bool.true) (hd : [p.y,p.mean,p.rstd,p.residualOut].Nodup)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    ((Forward.reference (Logical.algebra rest) p s).mem
      p.residualOut (s.pids 0*p.residualOutStride+i.1.val)).read .fp32 = Logical.input p s i.1.val := by
  have hzy : p.residualOut≠p.y := by simp_all [ne_comm]
  have hzm : p.residualOut≠p.mean := by simp_all [ne_comm]
  have hzr : p.residualOut≠p.rstd := by simp_all [ne_comm]
  unfold Forward.reference
  rw [Forward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hzy)]
  have stat : (Forward.statisticsState (Logical.algebra rest) p s).mem p.residualOut
      (s.pids 0*p.residualOutStride+i.1.val) =
      (Forward.residualState (Logical.algebra rest) p s).mem p.residualOut
        (s.pids 0*p.residualOutStride+i.1.val) := by
    cases hm : p.rms <;> simp [Forward.statisticsState,hm,State.write,hzm,hzr]
  rw [stat]
  simp only [Forward.residualState,hs,if_true]
  rw [Forward.scatter_read _ _ _ _ _ _ i hi]
  change Forward.z (Logical.algebra rest) p s i = _
  rw [Logical.z_value,if_pos hi]

 theorem forward_z_grid (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ)
    (rows row : Nat) (hrow : row<rows) (hs : p.columns≤p.residualOutStride)
    (stored : p.storeResidual=Bool.true) (hd : [p.y,p.mean,p.rstd,p.residualOut].Nodup)
    (hx : Host.ForwardRegionOutside p p.x) (hr : Host.ForwardRegionOutside p p.residual)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    ((Grid.reference (Forward.reference (Logical.algebra rest) p) s rows).mem
      p.residualOut (row*p.residualOutStride+i.1.val)).read .fp32 = z p s row i.1.val := by
  have hzy : p.residualOut≠p.y := by simp_all [ne_comm]
  have hzm : p.residualOut≠p.mean := by simp_all [ne_comm]
  have hzr : p.residualOut≠p.rstd := by simp_all [ne_comm]
  rw [Grid.readback _ s rows row hrow p.residualOut (row*p.residualOutStride+i.1.val) ?_]
  · trans Logical.input p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) i.1.val
    · simpa only [Grid.launch,if_true] using reference_residual rest p (Grid.launch s (Grid.reference (Forward.reference (Logical.algebra rest) p) s row).mem row) stored hd i hi
    · exact congrFun (forward_input rest p s row row hx hr) i.1.val
  · intro later hl hu mem
    apply Forward.reference_frame
    refine ⟨Or.inl hzy,fun _ => Or.inr ?_,fun _ => Or.inl hzm,Or.inl hzr⟩
    intro j hj
    exact Grid.row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs)
      row later (by omega)

#axiomsClean forward_y_grid
#axiomsClean forward_rstd_grid
#axiomsClean forward_mean_grid
#axiomsClean forward_z_grid
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
