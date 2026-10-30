import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_AffineSource

/-! Successful execution of the remaining source primitives. No inactive
coefficient lane is replaced by zero, and inactive lazy expression branches
need no initialized registers. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def maskTile (N block : Nat) : Tile .bool [block] :=
  Tile.vec (fun j => decide (j.val < N))

def coefficients (region : RegionName) (N : Nat) (s : BlockState) : Nat → ℝ :=
  fun col => if col < N then s.readMem region col else s.undef region col

theorem maskStmt_exec (N block : Nat) (s : BlockState)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val))) :
    stepStmt (maskStmt N block) s = some (s.setReg "mask" .bool [block] (maskTile N block)) := by
  simp [maskStmt, stepStmt, evalOp_lt, evalOp_ref, evalOp_constNat, hcols,
    maskTile, Tile.cop, ComparableDType.lt]
  rfl

theorem coefficientOp_eval (region : RegionName) (N block : Nat) (s : BlockState)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block)) :
    evalOp (coefficientOp region block) s =
      some (Execution.realTile (fun j => coefficients region N s j.val)) := by
  simp only [coefficientOp, evalOp, hcols, hmask, bind, Option.bind, pure]
  congr 1
  ext j
  by_cases hj : j.1.val < N
  all_goals simp [maskTile, coefficients, Execution.realTile, hj, Region.cast]

theorem storeStmt_exec (pointer name : RegName) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block))
    (hv : s.regs .real [block] name = some (Execution.realTile (fun j => values j.val))) :
    stepStmt (storeStmt pointer name block) s =
      some (Memory.scatter region start N block (fun col => MemCell.real (values col)) s) := by
  simp [storeStmt, stepStmt, evalOp, hptr, hcols, hmask, hv, Tile.ptrAdd,
    maskTile, Execution.realTile, Memory.scatter_fold, Memory.real_store]
  rfl

theorem eval_ite_bool (flag : Bool) {block : Nat} (a b : Op .real [block]) (s : BlockState) :
    evalOp (.ite (.constBool flag) a b) s = if flag then evalOp a s else evalOp b s := by
  simp [evalOp]

theorem eval_ite_not (flag : Bool) {block : Nat} (a b : Op .real [block]) (s : BlockState) :
    evalOp (.ite (.boolNot (.constBool flag)) a b) s =
      if !flag then evalOp a s else evalOp b s := by
  simp [evalOp, Tile.uop]

def normalizedValues (rms : Bool) (x : Nat → ℝ) (mean rstd : ℝ) : Nat → ℝ :=
  fun col => if rms then x col * rstd else (x col - mean) * rstd

theorem normalizedOp_eval (rms : Bool) (block : Nat) (x : Nat → ℝ)
    (mean rstd : ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Execution.realTile (fun j => x j.val)))
    (hm : rms = Bool.false → s.regs .real [] "mean" = some (Tile.scalar (some mean)))
    (hs : s.regs .real [] "rstd" = some (Tile.scalar (some rstd))) :
    evalOp (normalizedOp rms block) s =
      some (Execution.realTile (fun j => normalizedValues rms x mean rstd j.val)) := by
  rw [normalizedOp, eval_ite_not]
  cases hr : rms
  · simp [hr, evalOp_mul, evalOp_sub, evalOp_ref, hx, hm hr, hs,
      normalizedValues, Execution.realTile, Tile.bop, NumericDType.mul, NumericDType.sub]
  · simp [hr, evalOp_mul, evalOp_ref, hx, hs,
      normalizedValues, Execution.realTile, Tile.bop, NumericDType.mul]

def affineValues (bias : Bool) (xhat weight b : Nat → ℝ) : Nat → ℝ :=
  fun col => if bias then xhat col * weight col + b col else xhat col * weight col

theorem valueOp_eval (bias : Bool) (weightName biasName : RegName) (block : Nat)
    (xhat weight b : Nat → ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x_hat" = some (Execution.realTile (fun j => xhat j.val)))
    (hw : s.regs .real [block] weightName = some (Execution.realTile (fun j => weight j.val)))
    (hb : bias = Bool.true → s.regs .real [block] biasName =
      some (Execution.realTile (fun j => b j.val))) :
    evalOp (valueOp bias weightName biasName block) s =
      some (Execution.realTile (fun j => affineValues bias xhat weight b j.val)) := by
  rw [valueOp, eval_ite_bool]
  cases hf : bias
  · simp [hf, evalOp_mul, evalOp_ref, hx, hw,
      affineValues, Execution.realTile, Tile.bop, NumericDType.mul]
  · simp [hf, evalOp_mul, evalOp_add, evalOp_ref, hx, hw, hb hf,
      affineValues, Execution.realTile, Tile.bop, NumericDType.mul, NumericDType.add]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
