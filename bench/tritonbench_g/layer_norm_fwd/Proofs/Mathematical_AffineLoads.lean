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
  let t := s.setReg weightName .real [block]
    (Execution.realTile (fun j => coefficients weightRegion N s j.val))
  if bias then t.setReg biasName .real [block]
    (Execution.realTile (fun j => coefficients biasRegion N s j.val)) else t

theorem loadPhase_exec (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (hwc : "cols" ≠ weightName) (hwm : "mask" ≠ weightName)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block)) :
    stepStmts (loadPhase bias weightRegion biasRegion weightName biasName block) s =
      some (loadState bias weightRegion biasRegion weightName biasName N block s) := by
  have hw := stepStmt_assign_eq_some (name := weightName)
    (coefficientOp_eval weightRegion N block s hcols hmask)
  rw [loadPhase, stepStmts.cons_some hw, InputSuffix.exec_single, InputSuffix.step_if_const]
  cases bias
  · rfl
  · simp only [↓reduceIte]
    rw [InputSuffix.exec_single]
    have hb := stepStmt_assign_eq_some (name := biasName)
      (coefficientOp_eval biasRegion N block
        (s.setReg weightName .real [block]
          (Execution.realTile (fun j => coefficients weightRegion N s j.val)))
        (by simpa [hwc] using hcols) (by simpa [hwm] using hmask))
    simpa only [coefficients_setReg] using hb

theorem loadState_weight (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (hwb : weightName ≠ biasName) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs
      .real [block] weightName =
      some (Execution.realTile (fun j => coefficients weightRegion N s j.val)) := by
  cases bias <;> simp [loadState, hwb]

theorem loadState_bias (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState) (hb : bias = Bool.true) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs
      .real [block] biasName =
      some (Execution.realTile (fun j => coefficients biasRegion N s j.val)) := by
  simp [loadState, hb]

theorem loadState_regs (bias : Bool) (weightRegion biasRegion : RegionName)
    (weightName biasName : RegName) (N block : Nat) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hw : name ≠ weightName) (hb : name ≠ biasName) :
    (loadState bias weightRegion biasRegion weightName biasName N block s).regs dtype shape name =
      s.regs dtype shape name := by
  cases bias <;> simp [loadState, hw, hb]

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
