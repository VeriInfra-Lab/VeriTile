import bench.tritonbench_g.layer_norm_fwd.Proofs.AffineFirst

/-! The optional second affine output and composition of all remaining
source statements. Inactive Y1 pointers and B1 registers are unrestricted. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def secondLoadedState (f : Flags) (r : Regions) (N block : Nat) (s : BlockState) : BlockState :=
  loadState f.b1 r.weight1 r.bias1 "w1" "b1" N block s

def secondBodyState (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState) : BlockState :=
  outputState f.b1 "y1" r.y1 start N block xhat
    (coefficients r.weight1 N s) (coefficients r.bias1 N s)
    (secondLoadedState f r N block s)

def secondState (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState) : BlockState :=
  if f.w1 then secondBodyState f r start N block xhat s else s

theorem secondBody_exec (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState)
    (hptr : s.regs .ptr [] "Y1" = some (Tile.scalar (r.y1, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block))
    (hx : s.regs .real [block] "x_hat" = some (Execution.realTile (fun j => xhat j.val))) :
    stepStmts (loadPhase f.b1 r.weight1 r.bias1 "w1" "b1" block ++
      outputPhase f.b1 "w1" "b1" "y1" "Y1" block) s =
      some (secondBodyState f r start N block xhat s) := by
  have hl : stepStmts (loadPhase f.b1 r.weight1 r.bias1 "w1" "b1" block) s =
      some (secondLoadedState f r N block s) :=
    loadPhase_exec _ _ _ _ _ _ _ _ (by decide) (by decide) hcols hmask
  have ho := outputPhase_exec f.b1 "w1" "b1" "y1" "Y1" r.y1 start N block xhat
    (coefficients r.weight1 N s) (coefficients r.bias1 N s)
    (secondLoadedState f r N block s) (by decide) (by decide) (by decide)
    (by rw [secondLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
        exact hptr)
    (by rw [secondLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
        exact hcols)
    (by rw [secondLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
        exact hmask)
    (by rw [secondLoadedState, loadState_regs _ _ _ _ _ _ _ _ _ _ _ (by decide) (by decide)]
        exact hx)
    (loadState_weight _ _ _ _ _ _ _ _ (by decide))
    (loadState_bias _ _ _ _ _ _ _ _)
  rw [ProgramRewrite.append, hl, Option.bind_some]
  exact ho

theorem secondStmt_exec (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState)
    (hptr : f.w1 = Bool.true → s.regs .ptr [] "Y1" = some (Tile.scalar (r.y1, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block))
    (hx : s.regs .real [block] "x_hat" = some (Execution.realTile (fun j => xhat j.val))) :
    stepStmt (secondStmt f r block) s = some (secondState f r start N block xhat s) := by
  rw [secondStmt, InputSuffix.step_if_const]
  cases hf : f.w1
  · simp [secondState, hf]
  · simp only [↓reduceIte, secondState, hf]
    exact secondBody_exec f r start N block xhat s (hptr hf) hcols hmask hx

theorem secondBodyState_mem (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState) (hN : N ≤ block) (region : RegionName) (offset : Nat) :
    (secondBodyState f r start N block xhat s).mem region offset =
      if region = r.y1 ∧ Contract.inRow start N offset then
        MemCell.real (affineValues f.b1 xhat (coefficients r.weight1 N s)
          (coefficients r.bias1 N s) (offset-start)) else s.mem region offset := by
  rw [secondBodyState, outputState_mem _ _ _ _ _ _ _ _ _ _ hN]
  simp only [secondLoadedState, loadState_mem]

theorem secondState_mem (f : Flags) (r : Regions) (start N block : Nat)
    (xhat : Nat → ℝ) (s : BlockState) (hN : N ≤ block) (region : RegionName) (offset : Nat) :
    (secondState f r start N block xhat s).mem region offset =
      if f.w1 && decide (region = r.y1 ∧ Contract.inRow start N offset) then
        MemCell.real (affineValues f.b1 xhat (coefficients r.weight1 N s)
          (coefficients r.bias1 N s) (offset-start)) else s.mem region offset := by
  cases hf : f.w1 <;> simp [secondState, hf, secondBodyState_mem _ _ _ _ _ _ _ hN]

def state (f : Flags) (r : Regions) (start start1 N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState) : BlockState :=
  secondState f r start1 N block (normalizedValues f.rms x mean rstd)
    (firstState f r start N block x mean rstd s)

theorem phase_exec (f : Flags) (r : Regions) (start start1 N block : Nat)
    (x : Nat → ℝ) (mean rstd : ℝ) (s : BlockState)
    (hptr : s.regs .ptr [] "Y" = some (Tile.scalar (r.y, start)))
    (hptr1 : f.w1 = Bool.true → s.regs .ptr [] "Y1" = some (Tile.scalar (r.y1, start1)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile (fun j => x j.val)))
    (hm : f.rms = Bool.false → s.regs .real [] "mean" = some (Tile.scalar (some mean)))
    (hs : s.regs .real [] "rstd" = some (Tile.scalar (some rstd))) :
    stepStmts (phase f r N block) s = some (state f r start start1 N block x mean rstd s) := by
  have hfirst := firstPhase_exec f r start N block x mean rstd s hptr hcols hx hm hs
  have hsecond := secondStmt_exec f r start1 N block (normalizedValues f.rms x mean rstd)
    (firstState f r start N block x mean rstd s)
    (by intro hf
        rw [firstState_regs _ _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide) (by decide)]
        exact hptr1 hf)
    (by rw [firstState_regs _ _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide) (by decide)]
        exact hcols)
    (firstState_mask f r start N block x mean rstd s)
    (firstState_xhat f r start N block x mean rstd s)
  change stepStmts (firstPhase f r N block ++ [secondStmt f r block]) s = _
  rw [ProgramRewrite.append, hfirst, Option.bind_some, InputSuffix.exec_single]
  exact hsecond

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
