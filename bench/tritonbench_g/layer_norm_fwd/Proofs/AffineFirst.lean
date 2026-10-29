import bench.tritonbench_g.layer_norm_fwd.Proofs.AffineOutput

/-! Successful execution of the first affine output subtree: mask, weight,
optional bias, lazy normalization expression, affine expression and Y store. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def firstPhase (f : Flags) (r : Regions) (N block : Nat) : List Stmt :=
  [maskStmt N block] ++ loadPhase f.bias r.weight r.bias "w" "b" block ++
    [.assign .real [block] "x_hat" (normalizedOp f.rms block)] ++
    outputPhase f.bias "w" "b" "y" "Y" block

def maskedState (N block : Nat) (s : BlockState) : BlockState :=
  s.setReg "mask" .bool [block] (maskTile N block)

def firstLoadedState (f : Flags) (r : Regions) (N block : Nat) (s : BlockState) : BlockState :=
  loadState f.bias r.weight r.bias "w" "b" N block (maskedState N block s)

theorem firstLoadedState_regs (f : Flags) (r : Regions) (N block : Nat) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hm : name ≠ "mask") (hw : name ≠ "w") (hb : name ≠ "b") :
    (firstLoadedState f r N block s).regs dtype shape name = s.regs dtype shape name := by
  rw [firstLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ hw hb]
  simp [maskedState, hm]

theorem firstLoadedState_mask (f : Flags) (r : Regions) (N block : Nat) (s : BlockState) :
    (firstLoadedState f r N block s).regs .bool [block] "mask" = some (maskTile N block) := by
  rw [firstLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
  simp [maskedState]

def firstNormalizedState (f : Flags) (r : Regions) (N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) : BlockState :=
  (firstLoadedState f r N block s).setReg "x_hat" .real [block]
    (Execution.realTile (fun j => normalizedValues f.rms x mean rstd j.val))

theorem firstNormalizedState_regs (f : Flags) (r : Regions) (N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hm : name ≠ "mask") (hw : name ≠ "w") (hb : name ≠ "b") (hx : name ≠ "x_hat") :
    (firstNormalizedState f r N block x mean rstd s).regs dtype shape name =
      s.regs dtype shape name := by
  rw [firstNormalizedState, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hx]
  exact firstLoadedState_regs f r N block s dtype shape name hm hw hb

theorem firstNormalizedState_mask (f : Flags) (r : Regions) (N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) :
    (firstNormalizedState f r N block x mean rstd s).regs .bool [block] "mask" =
      some (maskTile N block) := by
  simpa [firstNormalizedState] using firstLoadedState_mask f r N block s

theorem firstNormalizedState_weight (f : Flags) (r : Regions) (N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) :
    (firstNormalizedState f r N block x mean rstd s).regs .real [block] "w" =
      some (Execution.realTile (fun j => coefficients r.weight N s j.val)) := by
  have hw := loadState_weight f.bias r.weight r.bias "w" "b" N block
    (maskedState N block s) (by decide)
  simpa [firstNormalizedState, firstLoadedState, maskedState] using hw

theorem firstNormalizedState_bias (f : Flags) (r : Regions) (N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) (hb : f.bias = Bool.true) :
    (firstNormalizedState f r N block x mean rstd s).regs .real [block] "b" =
      some (Execution.realTile (fun j => coefficients r.bias N s j.val)) := by
  have hv := loadState_bias f.bias r.weight r.bias "w" "b" N block (maskedState N block s) hb
  simpa [firstNormalizedState, firstLoadedState, maskedState] using hv

def firstState (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) : BlockState :=
  outputState f.bias "y" r.y start N block (normalizedValues f.rms x mean rstd)
    (coefficients r.weight N s) (coefficients r.bias N s)
    (firstNormalizedState f r N block x mean rstd s)

theorem firstPhase_exec (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState)
    (hptr : s.regs .ptr [] "Y" = some (Tile.scalar (r.y, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile (fun j => x j.val)))
    (hm : f.rms = Bool.false → s.regs .real [] "mean" = some (Tile.scalar (some mean)))
    (hs : s.regs .real [] "rstd" = some (Tile.scalar (some rstd))) :
    stepStmts (firstPhase f r N block) s = some (firstState f r start N block x mean rstd s) := by
  have hmask : stepStmts [maskStmt N block] s = some (maskedState N block s) := by
    rw [InputSuffix.exec_single]
    exact maskStmt_exec N block s hcols
  have hload : stepStmts (loadPhase f.bias r.weight r.bias "w" "b" block)
      (maskedState N block s) = some (firstLoadedState f r N block s) := by
    apply loadPhase_exec _ _ _ _ _ _ _ _ (by decide) (by decide)
    · simpa [maskedState] using hcols
    · simp [maskedState]
  have hnorm : stepStmts [.assign .real [block] "x_hat" (normalizedOp f.rms block)]
      (firstLoadedState f r N block s) = some (firstNormalizedState f r N block x mean rstd s) := by
    rw [InputSuffix.exec_single]
    apply stepStmt_assign_eq_some
    apply normalizedOp_eval
    · rw [firstLoadedState_regs _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]
      exact hx
    · intro hr
      rw [firstLoadedState_regs _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]
      exact hm hr
    · rw [firstLoadedState_regs _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]
      exact hs
  have hout := outputPhase_exec f.bias "w" "b" "y" "Y" r.y start N block
    (normalizedValues f.rms x mean rstd) (coefficients r.weight N s) (coefficients r.bias N s)
    (firstNormalizedState f r N block x mean rstd s) (by decide) (by decide) (by decide)
    (by rw [firstNormalizedState_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide)]
        exact hptr)
    (by rw [firstNormalizedState_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide)]
        exact hcols)
    (firstNormalizedState_mask f r N block x mean rstd s)
    (by simp [firstNormalizedState])
    (firstNormalizedState_weight f r N block x mean rstd s)
    (firstNormalizedState_bias f r N block x mean rstd s)
  simp only [firstPhase, ProgramRewrite.append, hmask, Option.bind_some, hload, hnorm]
  exact hout

theorem firstState_regs (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hm : name ≠ "mask") (hw : name ≠ "w") (hb : name ≠ "b")
    (hx : name ≠ "x_hat") (hy : name ≠ "y") :
    (firstState f r start N block x mean rstd s).regs dtype shape name =
      s.regs dtype shape name := by
  rw [firstState, outputState_regs _ _ _ _ _ _ _ _ _ _ _ _ _ hy]
  exact firstNormalizedState_regs f r N block x mean rstd s dtype shape name hm hw hb hx

theorem firstState_xhat (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) :
    (firstState f r start N block x mean rstd s).regs .real [block] "x_hat" =
      some (Execution.realTile (fun j => normalizedValues f.rms x mean rstd j.val)) := by
  rw [firstState, outputState_regs _ _ _ _ _ _ _ _ _ _ _ _ _ (by decide)]
  simp [firstNormalizedState]

theorem firstState_mask (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) :
    (firstState f r start N block x mean rstd s).regs .bool [block] "mask" =
      some (maskTile N block) := by
  rw [firstState, outputState_regs _ _ _ _ _ _ _ _ _ _ _ _ _ (by decide)]
  exact firstNormalizedState_mask f r N block x mean rstd s

theorem firstState_mem (f : Flags) (r : Regions) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState)
    (hN : N ≤ block) (region : RegionName) (offset : Nat) :
    (firstState f r start N block x mean rstd s).mem region offset =
      if region = r.y ∧ Contract.inRow start N offset then
        MemCell.real (affineValues f.bias (normalizedValues f.rms x mean rstd)
          (coefficients r.weight N s) (coefficients r.bias N s) (offset-start))
      else s.mem region offset := by
  rw [firstState, outputState_mem _ _ _ _ _ _ _ _ _ _ hN]
  simp only [firstNormalizedState, BlockState.setReg_mem, firstLoadedState, loadState_mem,
    maskedState]

theorem firstState_readsUnchanged (f : Flags) (r : Regions)
    (hsep : Domain.ReadWriteSeparated f r) (start N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) (hN : N ≤ block) :
    Domain.ReadsUnchanged f r (firstState f r start N block x mean rstd s) s := by
  intro region hr offset
  rw [firstState_mem _ _ _ _ _ _ _ _ _ hN]
  simp [hsep region hr r.y (by simp [Domain.writeRegion])]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
