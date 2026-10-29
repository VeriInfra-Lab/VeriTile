import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathRecomputed
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 1000000

def gridReads (p : BackwardParameters) := backwardReads p ++ [p.weight,p.bias]
def before (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ) (pid : Nat) :=
  Grid.launch s (Grid.reference (Backward.reference (Logical.algebra rest) p) s pid).mem pid

theorem grid_inputs (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ) (pid : Nat)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r) (r : RegionName) (hr : r∈gridReads p) :
    (before rest p s pid).mem r=s.mem r := by
  funext offset
  exact Host.backward_grid_frame _ _ _ _ _ _ (fresh r hr)

theorem grid_values (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ) (pid row col : Nat)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r) :
    rowDX p (before rest p s pid) row col=rowDX p s row col ∧
    rowY p (before rest p s pid) row col=rowY p s row col ∧
    Logical.rowDW p (before rest p s pid) row col=Logical.rowDW p s row col ∧
    Logical.rowDB p (before rest p s pid) row col=Logical.rowDB p s row col := by
  have hx := grid_inputs rest p s pid fresh p.x (by simp [gridReads,backwardReads])
  have hd := grid_inputs rest p s pid fresh p.dy (by simp [gridReads,backwardReads])
  have hm := grid_inputs rest p s pid fresh p.mean (by simp [gridReads,backwardReads])
  have hi := grid_inputs rest p s pid fresh p.rstd (by simp [gridReads,backwardReads])
  have hr := grid_inputs rest p s pid fresh p.dres (by simp [gridReads,backwardReads])
  have hw := grid_inputs rest p s pid fresh p.weight (by simp [gridReads,backwardReads])
  have hb := grid_inputs rest p s pid fresh p.bias (by simp [gridReads,backwardReads])
  simp only [rowDX,Logical.dx,rowY,readRow,Logical.rowDW,Logical.rowDB,hx,hd,hm,hi,hr,hw,hb,and_self]

theorem grid_dx (rest : Algebra ℝ) (fp : ForwardParameters) (p : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp p rows programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup) (hdy : p.recompute=Bool.true → p.dres≠p.y)
    (row : Nat) (hrow : row<rows) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs).mem
      p.dx (row*p.dxStride+i.1.val)=.mk .fp32 (rowDX p s row i.1.val) := by
  obtain ⟨pid,⟨hp,hlo,hhi⟩,_⟩ := Scope.row_ownership fp p rows programs row scope hrow
  have hf : ∀ r ∈ backwardReads p, Backward.RegionOutside p r := fun r hr => fresh r (by simp [gridReads,hr])
  have hb : p.columns≤p.block := by rw [scope.sameColumns,scope.sameBlock]; exact scope.blockCovers
  have hs : p.columns≤p.dxStride := by rw [scope.backwardStrides.2.2.2.1]
  have hxy : p.dx≠p.y := by simp_all
  have hxd : p.dx≠p.dresIn := by simp_all
  have hxw : p.dx≠p.dw := by simp_all
  have hxb : p.dx≠p.db := by simp_all
  rw [Grid.readback _ s programs pid hp p.dx (row*p.dxStride+i.1.val) ?_]
  · unfold Backward.reference
    rw [finish_region_frame _ _ _ _ _ _ hxw hxb]
    rw [loop_dx_inside rest p _ hf hb hs hxy hxd hdy row (by simpa [Backward.start,Grid.launch] using hlo)
      (by simpa [Backward.stop,Grid.launch] using hhi) i hi]
    exact congrArg (Cell.mk .fp32) (grid_values rest p s pid row i.1.val fresh).1
  · intro later hl hu mem
    unfold Backward.reference
    rw [finish_region_frame _ _ _ _ _ _ hxw hxb]
    apply loop_dx_outside rest p _ hf hb hs hxy hxd hdy row _ i hi
    left
    have hm := Nat.mul_le_mul_right p.rowsPerProgram (show pid+1≤later by omega)
    simp only [Backward.start,Grid.launch,if_true]
    exact lt_of_lt_of_le (lt_of_lt_of_le hhi (Nat.min_le_left _ _)) hm

theorem grid_y (rest : Algebra ℝ) (fp : ForwardParameters) (p : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp p rows programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup) (hy : p.recompute=Bool.true)
    (row : Nat) (hrow : row<rows) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs).mem
      p.y (row*p.yStride+i.1.val)=.mk .fp32 (rowY p s row i.1.val) := by
  obtain ⟨pid,⟨hp,hlo,hhi⟩,_⟩ := Scope.row_ownership fp p rows programs row scope hrow
  have hf : ∀ r ∈ backwardReads p, Backward.RegionOutside p r := fun r hr => fresh r (by simp [gridReads,hr])
  have hs : p.columns≤p.yStride := by rw [scope.backwardStrides.2.1,hy]; simp
  have hyx : p.y≠p.dx := by simp_all [ne_comm]
  have hyd : p.y≠p.dresIn := by simp_all
  have hyw : p.y≠p.dw := by simp_all [ne_comm]
  have hyb : p.y≠p.db := by simp_all [ne_comm]
  rw [Grid.readback _ s programs pid hp p.y (row*p.yStride+i.1.val) ?_]
  · unfold Backward.reference
    rw [finish_region_frame _ _ _ _ _ _ hyw hyb]
    rw [loop_y_inside rest p _ hf hy hs hyx hyd row (by simpa [Backward.start,Grid.launch] using hlo)
      (by simpa [Backward.stop,Grid.launch] using hhi) i hi]
    exact congrArg (Cell.mk .fp32) (grid_values rest p s pid row i.1.val fresh).2.1
  · intro later hl hu mem
    unfold Backward.reference
    rw [finish_region_frame _ _ _ _ _ _ hyw hyb]
    apply loop_y_outside rest p _ hf hy hs hyx hyd row _ i hi
    left
    have hm := Nat.mul_le_mul_right p.rowsPerProgram (show pid+1≤later by omega)
    simp only [Backward.start,Grid.launch,if_true]
    exact lt_of_lt_of_le (lt_of_lt_of_le hhi (Nat.min_le_left _ _)) hm

#axiomsClean grid_dx
#axiomsClean grid_y
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
