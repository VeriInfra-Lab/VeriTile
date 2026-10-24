import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoadCode

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem k_evaluates (p : Parameters) (s : BlockState) (ki : Nat)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int))) :
    evalOp (kOffsets p) s = some (Tile.vec fun k => (ki*p.BK+k.val : Nat)) := by
  simp [kOffsets, evalOp.eq_def, hk, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul,
    -Tile.scalar_eta]

theorem a_pointer_evaluates (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (ki : Nat)
    (hr : s.regs .int [p.BM] "offs_am" = some (Tile.vec fun i => (RealTiles.clamp p.M (row i) : Int)))
    (hk : s.regs .int [p.BK] "offs_k" = some (Tile.vec fun k => (ki*p.BK+k.val : Nat))) :
    evalOp (aPointers p) s = some (aAddresses p row ki) := by
  simp [aPointers, aAddresses, evalOp.eq_def, hr, hk, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.mat, Tile.bop, Tile.ptrAdd, Tile.expandDim,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Region.cast,
    -Tile.scalar_eta]
  funext i
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

theorem b_pointer_evaluates (p : Parameters) (s : BlockState)
    (column : Fin p.BN → Nat) (ki : Nat)
    (hc : s.regs .int [p.BN] "offs_bn" = some (Tile.vec fun j => (RealTiles.clamp p.N (column j) : Int)))
    (hk : s.regs .int [p.BK] "offs_k" = some (Tile.vec fun k => (ki*p.BK+k.val : Nat))) :
    evalOp (bPointers p) s = some (bAddresses p column ki) := by
  simp [bPointers, bAddresses, evalOp.eq_def, hc, hk, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.mat, Tile.bop, Tile.ptrAdd, Tile.expandDim,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Region.cast,
    -Tile.scalar_eta]
  funext i
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

theorem a_evaluates (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (ki : Nat)
    (hp : s.regs .ptr [p.BM,p.BK] "a_ptrs" = some (aAddresses p row ki))
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    evalOp (aLoad p) s = some (RealTiles.aValues p (memoryInputs p s) row ki) := by
  simp [aLoad, remaining, evalOp.eq_def, hp, hk, hl, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.mat, Tile.bop, Tile.cop, Tile.expandDim, Tile.remap,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    aAddresses, RealTiles.aValues, memoryInputs, signed_mask, -Tile.scalar_eta]
  funext i
  split_ifs <;> rfl

theorem b_evaluates (p : Parameters) (s : BlockState)
    (column : Fin p.BN → Nat) (ki : Nat)
    (hp : s.regs .ptr [p.BK,p.BN] "b_ptrs" = some (bAddresses p column ki))
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    evalOp (bLoad p) s = some (RealTiles.bValues p (memoryInputs p s) column ki) := by
  simp [bLoad, remaining, evalOp.eq_def, hp, hk, hl, bind, Option.bind,
    Tile.scalar, Tile.vec, Tile.mat, Tile.bop, Tile.cop, Tile.expandDim, Tile.remap,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    bAddresses, RealTiles.bValues, memoryInputs, signed_mask, -Tile.scalar_eta]
  funext i
  split_ifs <;> rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
