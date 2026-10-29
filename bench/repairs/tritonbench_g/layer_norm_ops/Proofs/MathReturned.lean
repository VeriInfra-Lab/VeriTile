import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathSaved
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 1500000

/-- The independent pre-call mathematical target is read through the actual
returned tensor regions in the final shared memory. Optional objects and the
dResidual/DX alias are part of the same postcondition. -/
structure ReturnPost (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows : Nat) (dwRegion dbRegion : RegionName)
    (f : Host.ForwardReturn ℝ) (b : Host.BackwardReturn ℝ) : Prop where
  descriptors : f.y=Host.matrix fp.y rows fp.columns ∧
    f.mean=(if fp.rms then none else some (Host.vector fp.mean rows)) ∧
    f.rstd=Host.vector fp.rstd rows ∧
    f.z=Host.matrix (if fp.storeResidual then fp.residualOut else fp.x) rows fp.columns ∧
    b.dx=Host.matrix bp.dx rows fp.columns ∧ b.dw=Host.vector dwRegion fp.columns ∧
    b.db=(if fp.hasBias then some (Host.vector dbRegion fp.columns) else none) ∧
    b.dResidual=(if fp.hasResidual then some b.dx else none) ∧
    b.recomputedY=(if bp.recompute then some (Host.matrix bp.y rows fp.columns) else none)
  forward : ∀ row, row<rows → ∀ col, col<fp.columns →
    (b.state.mem f.y.region (row*fp.columns+col)).read .fp32=y fp s row col ∧
    (b.state.mem f.z.region (row*fp.columns+col)).read .fp32=z fp s row col
  statistics : ∀ row, row<rows →
    (b.state.mem f.rstd.region row).read .fp32=inv fp s row ∧
    (fp.rms=Bool.false → ∀ tensor, f.mean=some tensor →
      (b.state.mem tensor.region row).read .fp32=mu fp s row)
  backward : ∀ row, row<rows → ∀ col, col<fp.columns →
    (b.state.mem b.dx.region (row*fp.columns+col)).read .fp32=dx fp bp s row col ∧
    (bp.recompute=Bool.true → ∀ tensor, b.recomputedY=some tensor →
      (b.state.mem tensor.region (row*fp.columns+col)).read .fp32=y fp s row col)
  affine : ∀ col, col<fp.columns →
    (b.state.mem b.dw.region col).read .fp32=dw fp bp s rows col ∧
    (fp.hasBias=Bool.true → ∀ tensor, b.db=some tensor →
      (b.state.mem tensor.region col).read .fp32=db bp s rows col)

theorem complete_math (rest : Algebra ℝ) (fp : ForwardParameters) (bp : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp bp rows programs)
    (pw pb : ReductionPlan.Plan programs) (dwRegion dbRegion : RegionName)
    (binding : RegionBinding fp bp dwRegion dbRegion) :
    ∃ result, Host.composed (Logical.algebra rest) scalar fp bp rows programs pw pb dwRegion dbRegion s=some result ∧
      ReturnPost fp bp s rows dwRegion dbRegion result.1 result.2 := by
  rw [Host.composed_complete]
  refine ⟨_,rfl,?_⟩
  let fs := forwardState rest fp s rows
  let bs := Grid.reference (Backward.reference (Logical.algebra rest) bp) fs programs
  have fresh : ∀ r ∈ gridReads bp, Backward.RegionOutside bp r := by
    intro r hr
    apply (binding.backwardFresh r ?_).1
    simp [gridReads,backwardReads] at hr
    simp
    tauto
  have distinct : [bp.dx,bp.dw,bp.db,bp.y,bp.dresIn].Nodup := by
    have h := binding.backwardDistinct
    exact List.Nodup.of_append_left (l₁:=[bp.dx,bp.dw,bp.db,bp.y,bp.dresIn]) (l₂:=[dwRegion,dbRegion]) h
  have finalFrame : ∀ r ∈ [bp.x,bp.weight,bp.bias,bp.dy,bp.dres,bp.mean,bp.rstd,
      fp.y,fp.mean,fp.rstd,fp.residualOut,fp.x,fp.residual], ∀ offset,
      (Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem r offset=fs.mem r offset := by
    intro r hr offset
    obtain ⟨hrg,hw,hb⟩ := binding.backwardFresh r hr
    rw [Host.finish_frame _ _ _ _ _ _ _ _ _ _ _ hw hb,Host.backward_grid_frame _ _ _ _ _ _ hrg]
  have noHostDX : bp.dx≠dwRegion ∧ bp.dx≠dbRegion := by have := binding.backwardDistinct; simp_all
  have noHostY : bp.y≠dwRegion ∧ bp.y≠dbRegion := by have := binding.backwardDistinct; simp_all
  have hout : dwRegion≠dbRegion := by have := binding.backwardDistinct; simp_all
  have houtb : bp.db≠dwRegion := by have := binding.backwardDistinct; simp_all
  have hdy : bp.recompute=Bool.true → bp.dres≠bp.y := (fresh bp.dres (by simp [gridReads,backwardReads])).2.2.2.1
  refine ⟨?_,?_,?_,?_,?_⟩
  · simp [Host.forwardFinish,Host.backwardFinish,scope.sameRows,scope.sameColumns,
      scope.sameBias,scope.noSeparateDres]
  · intro row hr col hc
    let i : TileIndex [fp.block] := (⟨col,Nat.lt_of_lt_of_le hc scope.blockCovers⟩,PUnit.unit)
    constructor
    · change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem fp.y _).read .fp32=_
      rw [finalFrame fp.y (by simp)]
      simpa only [scope.forwardStrides.2.1] using forward_y_grid rest fp s rows row hr scope.blockCovers
        (by rw [scope.forwardStrides.2.1]) binding.forwardDistinct
        (fun r hr => binding.forwardFresh r (by simp at hr ⊢; tauto)) i hc
    · change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem
        (if fp.storeResidual then fp.residualOut else fp.x) _).read .fp32=_
      rw [← scope.savedZ,finalFrame bp.x (by simp)]
      simpa only [readRow,scope.backwardStrides.1,scope.sameColumns] using
        saved_z rest fp bp s rows programs scope dwRegion dbRegion binding row col hr hc
  · intro row hr
    constructor
    · change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem fp.rstd row).read .fp32=_
      rw [finalFrame fp.rstd (by simp)]
      exact forward_rstd_grid rest fp s rows row hr scope.blockCovers binding.forwardDistinct
        (binding.forwardFresh fp.x (by simp)) (binding.forwardFresh fp.residual (by simp))
    · intro hm tensor ht
      have he : tensor=Host.vector fp.mean rows := by simpa [Host.forwardFinish,hm] using ht.symm
      subst tensor
      change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem fp.mean row).read .fp32=_
      rw [finalFrame fp.mean (by simp)]
      exact forward_mean_grid rest fp s rows row hr scope.blockCovers binding.forwardDistinct hm
        (binding.forwardFresh fp.x (by simp)) (binding.forwardFresh fp.residual (by simp))
  · intro row hr col hc
    have hbc : col<bp.columns := by rw [scope.sameColumns]; exact hc
    have hblock : col<bp.block := by rw [scope.sameBlock]; exact Nat.lt_of_lt_of_le hc scope.blockCovers
    let i : TileIndex [bp.block] := (⟨col,hblock⟩,PUnit.unit)
    have values := produced_values rest fp bp s rows programs scope dwRegion dbRegion binding row col hr hc
    constructor
    · change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem bp.dx _).read .fp32=_
      rw [Host.finish_frame _ _ _ _ _ _ _ _ _ _ _ noHostDX.1 noHostDX.2]
      have cell := grid_dx rest fp bp fs rows programs scope fresh distinct hdy row hr i hbc
      rw [scope.backwardStrides.2.2.2.1,scope.sameColumns] at cell
      rw [cell]
      exact values.1
    · intro hy tensor ht
      have he : tensor=Host.matrix bp.y bp.rows bp.columns := by simpa [Host.backwardFinish,hy] using ht.symm
      subst tensor
      change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem bp.y _).read .fp32=_
      rw [Host.finish_frame _ _ _ _ _ _ _ _ _ _ _ noHostY.1 noHostY.2]
      have cell := grid_y rest fp bp fs rows programs scope fresh distinct hy row hr i hbc
      simp only [scope.backwardStrides.2.1,hy,if_true,scope.sameColumns] at cell
      rw [cell]
      exact values.2.1
  · intro col hc
    have hbc : col<bp.columns := by rw [scope.sameColumns]; exact hc
    constructor
    · change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem dwRegion col).read .fp32=_
      rw [return_dw rest fp bp fs rows programs scope fresh distinct pw pb dwRegion dbRegion fp.hasResidual hout ⟨col,hbc⟩]
      change (∑ row ∈ Finset.range rows, Logical.rowDW bp fs row col)=_
      apply Finset.sum_congr rfl
      intro row hr
      exact (produced_values rest fp bp s rows programs scope dwRegion dbRegion binding row col (Finset.mem_range.mp hr) hc).2.2.1
    · intro hb tensor ht
      have hbb : bp.hasBias=Bool.true := scope.sameBias.trans hb
      have he : tensor=Host.vector dbRegion bp.columns := by simpa [Host.backwardFinish,hbb] using ht.symm
      subst tensor
      change ((Host.backwardFinish scalar bp programs pw pb dwRegion dbRegion fp.hasResidual bs).state.mem dbRegion col).read .fp32=_
      rw [return_db rest fp bp fs rows programs scope fresh distinct hbb pw pb dwRegion dbRegion fp.hasResidual houtb ⟨col,hbc⟩]
      change (∑ row ∈ Finset.range rows, Logical.rowDB bp fs row col)=_
      apply Finset.sum_congr rfl
      intro row hr
      exact (produced_values rest fp bp s rows programs scope dwRegion dbRegion binding row col (Finset.mem_range.mp hr) hc).2.2.2

#specNonCircular ReturnPost avoiding [forwardKernel,backwardKernel,Host.composed]
#axiomsClean complete_math
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
