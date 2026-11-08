import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_AffinePrimitives

/-! Both affine coefficient-loading phases, including optional biases and
the exact inactive-lane undef values. No coefficient load writes memory. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

@[simp] theorem coefficients_setReg (region : RegionName) (N : Nat) (s : BlockState)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    coefficients region N (s.setReg name dtype shape v) = coefficients region N s := rfl

def loadState (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) : BlockState :=
  let w := Execution.realTile (fun j : Fin block => coefficients weightRegion N s j.val)
  let b := Execution.realTile (fun j : Fin block => coefficients biasRegion N s j.val)
  let t := (s.setReg (privateWeight weightName) .real [block] w).setReg weightName .real [block] w
  if bias then (t.setReg (privateBias biasName) .real [block] b).setReg biasName .real [block] b else t

theorem loadPhase_exec (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (hwc : "cols" ≠ weightName) (hwm : "mask" ≠ weightName)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block)) :
    stepStmts (loadPhase bias weightRegion biasRegion weightName biasName block) s =
      some (loadState bias weightRegion biasRegion weightName biasName N block s) := by
  let w := Execution.realTile (fun j : Fin block => coefficients weightRegion N s j.val)
  let b := Execution.realTile (fun j : Fin block => coefficients biasRegion N s j.val)
  let loadedW := s.setReg (privateWeight weightName) .real [block] w
  let copiedW := loadedW.setReg weightName .real [block] w
  let loadedB := copiedW.setReg (privateBias biasName) .real [block] b
  let copiedB := loadedB.setReg biasName .real [block] b
  have hw : stepStmt (.assign .real [block] (privateWeight weightName)
      (coefficientOp weightRegion block)) s=some loadedW :=
    stepStmt_assign_eq_some (coefficientOp_eval weightRegion N block s hcols hmask)
  have hcw : stepStmt (.assign .real [block] weightName
      (.ref .real [block] (privateWeight weightName))) loadedW=some copiedW := by
    simp [loadedW,copiedW,stepStmt,evalOp.eq_def]
  have hb : stepStmt (.assign .real [block] (privateBias biasName)
      (coefficientOp biasRegion block)) copiedW=some loadedB := by
    simpa [loadedB,b,copiedW,loadedW] using
      stepStmt_assign_eq_some (name := privateBias biasName)
        (coefficientOp_eval biasRegion N block copiedW
          (by by_cases h : weightName="w" <;> simpa [copiedW,loadedW,privateWeight,hwc,h] using hcols)
          (by by_cases h : weightName="w" <;> simpa [copiedW,loadedW,privateWeight,hwm,h] using hmask))
  have hcb : stepStmt (.assign .real [block] biasName
      (.ref .real [block] (privateBias biasName))) loadedB=some copiedB := by
    simp [loadedB,copiedB,stepStmt,evalOp.eq_def]
  rw [loadPhase,stepStmts.cons_some hw,stepStmts.cons_some hcw,
    InputSuffix.exec_single,InputSuffix.step_if_const]
  cases bias
  · rfl
  · simp only [↓reduceIte]
    rw [stepStmts.cons_some hb,stepStmts.cons_some hcb,stepStmts.nil]
    rfl

theorem loadState_weight (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (hwb : weightName ≠ biasName)
    (hwp : weightName ≠ privateBias biasName := by first | assumption | decide) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs
      .real [block] weightName =
      some (Execution.realTile (fun j => coefficients weightRegion N s j.val)) := by
  cases bias <;> simp [loadState, hwb,hwp]

theorem loadState_bias (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) (hb : bias = Bool.true) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs
      .real [block] biasName =
      some (Execution.realTile (fun j => coefficients biasRegion N s j.val)) := by
  simp [loadState, hb]

theorem loadState_regs (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hw : name ≠ weightName) (hb : name ≠ biasName)
    (hpw : name ≠ privateWeight weightName := by first | assumption | decide)
    (hpb : name ≠ privateBias biasName := by first | assumption | decide) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs dtype shape name =
      s.regs dtype shape name := by
  cases bias <;> simp [loadState, hw, hb,hpw,hpb]

theorem loadState_mem (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).mem = s.mem := by
  cases bias <;> rfl

theorem loadState_coefficients (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) (region : RegionName) :
    coefficients region N (loadState bias weightRegion biasRegion weightName biasName N block s) =
      coefficients region N s := by
  cases bias <;> rfl

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
