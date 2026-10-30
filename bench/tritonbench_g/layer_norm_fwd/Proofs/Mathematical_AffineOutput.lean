import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_AffineLoads
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_InputDomain

/-! A complete affine value assignment and masked output write. It is used
for both Y and the optional Y1, with exact tagged memory and input preservation. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def outputState (bias : Bool) (outputName : RegName) (region : RegionName)
    (start N block : Nat) (xhat weight b : Nat → ℝ) (s : BlockState) : BlockState :=
  Memory.scatter region start N block
    (fun col => MemCell.real (affineValues bias xhat weight b col))
    (s.setReg outputName .real [block]
      (Execution.realTile (fun j => affineValues bias xhat weight b j.val)))

theorem outputPhase_exec (bias : Bool) (weightName biasName outputName pointer : RegName)
    (region : RegionName) (start N block : Nat) (xhat weight b : Nat → ℝ) (s : BlockState)
    (hpc : pointer ≠ outputName) (hcc : "cols" ≠ outputName) (hmc : "mask" ≠ outputName)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (maskTile N block))
    (hx : s.regs .real [block] "x_hat" = some (Execution.realTile (fun j => xhat j.val)))
    (hw : s.regs .real [block] weightName = some (Execution.realTile (fun j => weight j.val)))
    (hb : bias = Bool.true → s.regs .real [block] biasName =
      some (Execution.realTile (fun j => b j.val))) :
    stepStmts (outputPhase bias weightName biasName outputName pointer block) s =
      some (outputState bias outputName region start N block xhat weight b s) := by
  have hv := stepStmt_assign_eq_some (name := outputName)
    (valueOp_eval bias weightName biasName block xhat weight b s hx hw hb)
  rw [outputPhase, stepStmts.cons_some hv, InputSuffix.exec_single]
  apply storeStmt_exec
  · simpa [hpc] using hptr
  · simpa [hcc] using hcols
  · simpa [hmc] using hmask
  · simp

theorem outputState_regs (bias : Bool) (outputName : RegName) (region : RegionName)
    (start N block : Nat) (xhat weight b : Nat → ℝ) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hne : name ≠ outputName) :
    (outputState bias outputName region start N block xhat weight b s).regs dtype shape name =
      s.regs dtype shape name := by
  simp [outputState, hne]

theorem outputState_mem (bias : Bool) (outputName : RegName) (region : RegionName)
    (start N block : Nat) (xhat weight b : Nat → ℝ) (s : BlockState)
    (hN : N ≤ block) (other : RegionName) (offset : Nat) :
    (outputState bias outputName region start N block xhat weight b s).mem other offset =
      if other = region ∧ Contract.inRow start N offset then
        MemCell.real (affineValues bias xhat weight b (offset-start)) else s.mem other offset := by
  simp [outputState, Memory.scatter_cell _ _ _ _ hN]

theorem outputState_readsUnchanged (f : Flags) (r : Regions)
    (hsep : Domain.ReadWriteSeparated f r) (bias : Bool) (outputName : RegName)
    (region : RegionName) (hw : Domain.writeRegion f r region)
    (start N block : Nat) (xhat weight b : Nat → ℝ) (s : BlockState) (hN : N ≤ block) :
    Domain.ReadsUnchanged f r
      (outputState bias outputName region start N block xhat weight b s) s := by
  intro other hr offset
  rw [outputState_mem _ _ _ _ _ _ _ _ _ _ hN]
  simp [hsep other hr region hw]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
