import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathPartials
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 1000000

def forwardState (rest : Algebra ℝ) (p : ForwardParameters) (s : State ℝ) (rows : Nat) :=
  Grid.reference (Forward.reference (Logical.algebra rest) p) s rows

theorem saved_z (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp bp rows programs)
    (dw db : RegionName) (binding : RegionBinding fp bp dw db)
    (row col : Nat) (hr : row<rows) (hc : col<fp.columns) :
    readRow (forwardState rest fp s rows) bp.x bp.xStride row col=z fp s row col := by
  have hx := binding.forwardFresh fp.x (by simp)
  have hres := binding.forwardFresh fp.residual (by simp)
  rw [scope.savedZ,scope.backwardStrides.1,scope.sameColumns]
  cases hs : fp.storeResidual
  · have hn : fp.hasResidual=Bool.false := scope.storeResidual.symm.trans hs
    simp only [hs,Bool.false_eq_true,if_false,readRow,forwardState,z,hn,add_zero]
    rw [Host.forward_grid_frame _ _ _ _ _ _ hx,scope.forwardStrides.1]
  · simp only [hs,if_true,readRow]
    have hstride : fp.residualOutStride=fp.columns := by rw [scope.forwardStrides.2.2.2,hs]; rfl
    let i : TileIndex [fp.block] := (⟨col,Nat.lt_of_lt_of_le hc scope.blockCovers⟩,PUnit.unit)
    simpa only [hstride] using forward_z_grid rest fp s rows row hr (by omega) hs binding.forwardDistinct hx hres i hc

theorem saved_mean (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp bp rows programs)
    (dw db : RegionName) (binding : RegionBinding fp bp dw db)
    (row : Nat) (hr : row<rows) :
    (if bp.rms then 0 else ((forwardState rest fp s rows).mem bp.mean (row*bp.meanStride)).read .fp32)=mu fp s row := by
  rw [scope.sameRms]
  cases hm : fp.rms
  · simp only [Bool.false_eq_true,if_false]
    rw [scope.savedMean,scope.savedStrides.1,scope.sameRms,hm]
    simp only [Bool.false_eq_true,if_false,Nat.mul_one]
    exact forward_mean_grid rest fp s rows row hr scope.blockCovers binding.forwardDistinct hm
      (binding.forwardFresh fp.x (by simp)) (binding.forwardFresh fp.residual (by simp))
  · simp [mu,Logical.mean,hm]

theorem saved_inv (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp bp rows programs)
    (dw db : RegionName) (binding : RegionBinding fp bp dw db)
    (row : Nat) (hr : row<rows) :
    ((forwardState rest fp s rows).mem bp.rstd (row*bp.rstdStride)).read .fp32=inv fp s row := by
  rw [scope.savedRstd,scope.savedStrides.2,Nat.mul_one]
  exact forward_rstd_grid rest fp s rows row hr scope.blockCovers binding.forwardDistinct
    (binding.forwardFresh fp.x (by simp)) (binding.forwardFresh fp.residual (by simp))

theorem forward_read (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows : Nat) (dw db : RegionName) (binding : RegionBinding fp bp dw db)
    (r : RegionName) (hr : r∈[fp.x,fp.residual,fp.weight,fp.bias,bp.dy,bp.dres]) :
    (forwardState rest fp s rows).mem r=s.mem r := by
  funext o
  exact Host.forward_grid_frame _ _ _ _ _ _ (binding.forwardFresh r hr)

theorem produced_values (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp bp rows programs)
    (dw db : RegionName) (binding : RegionBinding fp bp dw db)
    (row col : Nat) (hr : row<rows) (hc : col<fp.columns) :
    rowDX bp (forwardState rest fp s rows) row col=dx fp bp s row col ∧
    rowY bp (forwardState rest fp s rows) row col=y fp s row col ∧
    Logical.rowDW bp (forwardState rest fp s rows) row col=
      dy bp s row col*Logical.normalized fp.columns fp.rms fp.epsilon (z fp s row) col ∧
    Logical.rowDB bp (forwardState rest fp s rows) row col=dy bp s row col := by
  have hz := saved_z rest fp bp s rows programs scope dw db binding row
  have hm := saved_mean rest fp bp s rows programs scope dw db binding row hr
  have hi := saved_inv rest fp bp s rows programs scope dw db binding row hr
  have hw := forward_read rest fp bp s rows dw db binding fp.weight (by simp)
  have hb := forward_read rest fp bp s rows dw db binding fp.bias (by simp)
  have hd := forward_read rest fp bp s rows dw db binding bp.dy (by simp)
  have hres := forward_read rest fp bp s rows dw db binding bp.dres (by simp)
  have hmean : (if fp.rms then 0 else mu fp s row)=mu fp s row := by cases h : fp.rms <;> simp [mu,Logical.mean,h]
  have hs1 : LayerNormLiger.Logical.sum bp.columns (fun j =>
      (readRow (forwardState rest fp s rows) bp.x bp.xStride row j -
        if bp.rms then 0 else ((forwardState rest fp s rows).mem bp.mean (row*bp.meanStride)).read .fp32) *
        ((forwardState rest fp s rows).mem bp.rstd (row*bp.rstdStride)).read .fp32 *
        (((forwardState rest fp s rows).mem bp.weight j).read .fp32 *
          readRow (forwardState rest fp s rows) bp.dy bp.dyStride row j)) =
      LayerNormLiger.Logical.sum fp.columns (fun j =>
        (z fp s row j-mu fp s row)*inv fp s row*(weight fp s j*dy bp s row j)) := by
    rw [scope.sameColumns]
    apply Finset.sum_congr rfl
    intro j hj
    dsimp only
    rw [hz j hr (Finset.mem_range.mp hj),hm,hi,scope.sameWeight,hw]
    simp [readRow,weight,dy,hd]
  have hs2 : LayerNormLiger.Logical.sum bp.columns (fun j =>
      ((forwardState rest fp s rows).mem bp.weight j).read .fp32 *
        readRow (forwardState rest fp s rows) bp.dy bp.dyStride row j)=
      LayerNormLiger.Logical.sum fp.columns (fun j => weight fp s j*dy bp s row j) := by
    rw [scope.sameColumns,scope.sameWeight,hw]
    simp [readRow,weight,dy,hd]
  refine ⟨?_,?_,?_,?_⟩
  · unfold rowDX Logical.dx
    dsimp only
    rw [hs1,hs2,hm,hi,hz col hr hc,scope.sameWeight,hw]
    simp only [readRow,hd,hres,scope.sameColumns,scope.sameRms]
    simp only [dx,Logical.dx,weight,dy,dres,hmean]
  · unfold rowY
    rw [hz col hr hc,hm,hi,scope.sameWeight,hw,scope.sameBiasRegion,hb,scope.sameBias]
    rfl
  · change readRow (forwardState rest fp s rows) bp.dy bp.dyStride row col *
      ((readRow (forwardState rest fp s rows) bp.x bp.xStride row col -
        if bp.rms then 0 else ((forwardState rest fp s rows).mem bp.mean (row*bp.meanStride)).read .fp32)*
        ((forwardState rest fp s rows).mem bp.rstd (row*bp.rstdStride)).read .fp32)=_
    rw [hz col hr hc,hm,hi]
    simp only [readRow,hd,dy,Logical.normalized,mu,inv]
  · simp [Logical.rowDB,dy,hd]

#axiomsClean saved_z
#axiomsClean saved_mean
#axiomsClean saved_inv
#axiomsClean produced_values
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
