import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FlatDomain
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Real

/-! Active-address safety for the actual source memory expressions. These
lemmas use declared finite bounds, not a kernel safety assumption. Numerical
values remain abstract while row, column and pointer registers are tracked. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem singleton (bounds : RegionBounds) (st : Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds [st] s ↔ Stmt.TraceSafe bounds st s := by
  rw [Stmt.TraceSafeList]
  cases stepStmt st s <;> simp [Stmt.TraceSafeList]

theorem if_const (bounds : RegionBounds) (flag : Bool) (body : List Stmt) (s : BlockState) :
    Stmt.TraceSafe bounds (.ifThen (.constBool flag) body) s ↔
      (flag = Bool.true → Stmt.TraceSafeList bounds body s) := by
  cases flag <;> simp [Stmt.TraceSafe, Op.SafeAt, evalOp]

theorem rowLoadOp_safe (pointer : RegName) (region : RegionName) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ col, col < N → start+col < bounds region) :
    (InputSuffix.rowLoadOp pointer N block).SafeAt bounds s := by
  simp [InputSuffix.rowLoadOp, Op.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    MaskOpt.Active, evalOp, hptr, hcols, Tile.ptrAdd, Tile.cop, ComparableDType.lt]
  exact fun j hj => hb j.val hj

theorem coefficientOp_safe (region : RegionName) (N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (Affine.maskTile N block))
    (hb : ∀ col, col < N → col < bounds region) :
    (Affine.coefficientOp region block).SafeAt bounds s := by
  simp [Affine.coefficientOp, Op.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, evalOp, hcols, hmask, Affine.maskTile, Region.cast]
  exact fun j hj => hb j.val hj

theorem rowStore_safe (pointer name : RegName) (region : RegionName) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ col, col < N → start+col < bounds region) :
    Stmt.TraceSafe bounds (InputSuffix.realStoreStmt pointer name N block) s := by
  simp [InputSuffix.realStoreStmt, Stmt.TraceSafe, Op.SafeAt, MemAccess.SafeAt,
    MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.Active, evalOp, hptr, hcols,
    Tile.ptrAdd, Tile.cop, ComparableDType.lt]
  exact fun j hj => hb j.val hj

theorem affineStore_safe (pointer name : RegName) (region : RegionName) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmask : s.regs .bool [block] "mask" = some (Affine.maskTile N block))
    (hb : ∀ col, col < N → start+col < bounds region) :
    Stmt.TraceSafe bounds (Affine.storeStmt pointer name block) s := by
  simp [Affine.storeStmt, Stmt.TraceSafe, Op.SafeAt, MemAccess.SafeAt,
    MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.Active, evalOp, hptr, hcols,
    hmask, Affine.maskTile, Tile.ptrAdd]
  exact fun j hj => hb j.val hj

theorem scalarStore_safe (region : RegionName) (name : RegName) (row : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) (hb : row < bounds region) :
    Stmt.TraceSafe bounds (Normalization.scalarStore region name) s := by
  simp [Normalization.scalarStore, Stmt.TraceSafe, Op.SafeAt, MemAccess.SafeAt,
    MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, evalOp, hrow, hb, Region.cast]

theorem firstRandomOp_safe (seeds : RegionName) (row block : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) (hb : row < bounds seeds) :
    (Execution.randomOp seeds block).SafeAt bounds s := by
  simp [Execution.randomOp, Op.intToUnsigned, Op.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, evalOp, hrow, hb, Region.cast]

theorem secondRandomOp_safe (seeds : RegionName) (M row block : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) (hb : M+row < bounds seeds) :
    (InputSuffix.secondRandomOp seeds M block).SafeAt bounds s := by
  simp [InputSuffix.secondRandomOp, InputSuffix.secondRowOp, Op.intToUnsigned,
    Op.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, evalOp, hrow, Tile.bop, NumericDType.add, hb, Region.cast]

theorem firstMask_safe (mask : RegionName) (row N block : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ col, col < N → row*N+col < bounds mask) :
    Stmt.TraceSafe bounds (Execution.dropoutMaskStmt mask N block) s := by
  simp [Execution.dropoutMaskStmt, Stmt.TraceSafe, Op.SafeAt, MemAccess.SafeAt,
    MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
    evalOp, hrow, hcols, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, Region.cast]
  exact fun j hj => hb j.val hj

theorem secondMask_safe (mask : RegionName) (M row N block : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : ∀ col, col < N → (M+row)*N+col < bounds mask) :
    Stmt.TraceSafe bounds (InputSuffix.maskStmt mask M N block) s := by
  simp [InputSuffix.maskStmt, InputSuffix.secondRowOp, Stmt.TraceSafe, Op.SafeAt,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, evalOp, hrow, hcols, Tile.bop, Tile.cop, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Region.cast]
  exact fun j hj => hb j.val hj

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
