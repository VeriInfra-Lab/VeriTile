import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyFirstStream

/-! Finite safety of all affine loads and stores, including the actual
no-other masks. Only active coefficient lanes require allocated memory. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem coefficientPhase_safe (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) (bounds : RegionBounds)
    (hwc : "cols" ≠ weightName) (hwm : "mask" ≠ weightName)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (Affine.maskTile N block))
    (hw : ∀ col, col < N → col < bounds weightRegion)
    (hb : bias = Bool.true → ∀ col, col < N → col < bounds biasRegion) :
    Stmt.TraceSafeList bounds (Affine.loadPhase bias weightRegion biasRegion weightName biasName block) s := by
  unfold Affine.loadPhase
  apply TraceProofs.assign_cons (coefficientOp_safe weightRegion N block s bounds hcols hmask hw)
  intro value
  rw [singleton, if_const]
  intro hf
  rw [singleton]
  simpa only [Stmt.TraceSafe] using
    coefficientOp_safe biasRegion N block (s.setReg weightName .real [block] value) bounds
      (by simpa [hwc] using hcols) (by simpa [hwm] using hmask) (hb hf)

theorem affineOutput_safe (bias : Bool) (weightName biasName outputName pointer : RegName)
    (region : RegionName) (start N block : Nat) (s : BlockState) (bounds : RegionBounds)
    (hop : pointer ≠ outputName) (hoc : "cols" ≠ outputName) (hom : "mask" ≠ outputName)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (Affine.maskTile N block))
    (hb : ∀ col, col < N → start+col < bounds region) :
    Stmt.TraceSafeList bounds (Affine.outputPhase bias weightName biasName outputName pointer block) s := by
  unfold Affine.outputPhase
  apply TraceProofs.assign_cons
  · simp [Affine.valueOp, Op.SafeAt]
  intro value
  rw [singleton]
  exact affineStore_safe pointer outputName region start N block _ bounds
    (by simpa [hop] using hptr) (by simpa [hoc] using hcols)
    (by simpa [hom] using hmask) hb

theorem affineFirst_safe (f : Flags) (r : Regions) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : s.regs .ptr [] "Y" = some (Tile.scalar (r.y, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hw : ∀ col, col < N → col < bounds r.weight)
    (hb : f.bias = Bool.true → ∀ col, col < N → col < bounds r.bias)
    (hy : ∀ col, col < N → start+col < bounds r.y) :
    Stmt.TraceSafeList bounds (Affine.firstPhase f r N block) s := by
  simp only [Affine.firstPhase, List.append_assoc]
  apply append_after bounds _ _ s (Affine.maskedState N block s)
  · rw [singleton]
    simp [Affine.maskStmt, Stmt.TraceSafe, Op.SafeAt]
  · rw [InputSuffix.exec_single]
    exact Affine.maskStmt_exec N block s hcols
  apply append_after bounds _ _ _ (Affine.firstLoadedState f r N block s)
  · apply coefficientPhase_safe f.bias r.weight r.bias "w" "b" N block _ bounds
      (by decide) (by decide) _ _ hw hb
    · simpa [Affine.maskedState] using hcols
    · simp [Affine.maskedState]
  · apply Affine.loadPhase_exec _ _ _ _ _ _ _ _ (by decide) (by decide)
    · simpa [Affine.maskedState] using hcols
    · simp [Affine.maskedState]
  simp only [List.singleton_append]
  apply TraceProofs.assign_cons
  · simp [Affine.normalizedOp, Op.SafeAt]
  intro value
  apply affineOutput_safe f.bias "w" "b" "y" "Y" r.y start N block _ bounds
    (by decide) (by decide) (by decide)
  · have hp := (Affine.firstLoadedState_regs f r N block s .ptr [] "Y"
        (by decide) (by decide) (by decide)).trans hptr
    simpa using hp
  · have hc := (Affine.firstLoadedState_regs f r N block s .nat [block] "cols"
        (by decide) (by decide) (by decide)).trans hcols
    simpa using hc
  · simpa using Affine.firstLoadedState_mask f r N block s
  · exact hy

theorem affineSecond_safe (f : Flags) (r : Regions) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : f.w1 = Bool.true → s.regs .ptr [] "Y1" = some (Tile.scalar (r.y1, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (Affine.maskTile N block))
    (hw : f.w1 = Bool.true → ∀ col, col < N → col < bounds r.weight1)
    (hb : f.w1 = Bool.true → f.b1 = Bool.true → ∀ col, col < N → col < bounds r.bias1)
    (hy : f.w1 = Bool.true → ∀ col, col < N → start+col < bounds r.y1) :
    Stmt.TraceSafe bounds (Affine.secondStmt f r block) s := by
  rw [Affine.secondStmt, if_const]
  intro hf
  apply append_after bounds _ _ s (Affine.loadState f.b1 r.weight1 r.bias1 "w1" "b1" N block s)
  · exact coefficientPhase_safe f.b1 r.weight1 r.bias1 "w1" "b1" N block s bounds
      (by decide) (by decide) hcols hmask (hw hf) (hb hf)
  · exact Affine.loadPhase_exec f.b1 r.weight1 r.bias1 "w1" "b1" N block s
      (by decide) (by decide) hcols hmask
  apply affineOutput_safe f.b1 "w1" "b1" "y1" "Y1" r.y1 start N block _ bounds
    (by decide) (by decide) (by decide)
  · rw [Affine.loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact hptr hf
  · rw [Affine.loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact hcols
  · rw [Affine.loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
    exact hmask
  · exact hy hf

theorem affine_safe (f : Flags) (r : Regions) (stride : Strides) (M row N block : Nat)
    (x : Nat → ℝ) (mean std : ℝ) (s : BlockState) (bounds : RegionBounds)
    (hptr : s.regs .ptr [] "Y" = some (Tile.scalar (r.y, row*stride.y)))
    (hptr1 : f.w1 = Bool.true → s.regs .ptr [] "Y1" = some (Tile.scalar (r.y1, row*stride.y1)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile (fun j => x j.val)))
    (hm : f.rms = Bool.false → s.regs .real [] "mean" = some (Tile.scalar (some mean)))
    (hs : s.regs .real [] "rstd" = some (Tile.scalar (some std)))
    (hb : FlatDomain.RowBounds f r stride M N row bounds) :
    Stmt.TraceSafeList bounds (Affine.phase f r N block) s := by
  change Stmt.TraceSafeList bounds (Affine.firstPhase f r N block ++ [Affine.secondStmt f r block]) s
  apply append_after bounds _ _ s (Affine.firstState f r (row*stride.y) N block x mean std s)
  · exact affineFirst_safe f r (row*stride.y) N block s bounds hptr hcols hb.weight hb.bias hb.y
  · exact Affine.firstPhase_exec f r (row*stride.y) N block x mean std s hptr hcols hx hm hs
  rw [singleton]
  apply affineSecond_safe f r (row*stride.y1) N block _ bounds
  · intro hf
    rw [Affine.firstState_regs _ _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hptr1 hf
  · rw [Affine.firstState_regs _ _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hcols
  · exact Affine.firstState_mask f r (row*stride.y) N block x mean std s
  · exact hb.weight1
  · exact hb.bias1
  · exact hb.y1

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
