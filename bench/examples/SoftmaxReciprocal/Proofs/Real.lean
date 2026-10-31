import bench.examples.SoftmaxReciprocal.Kernels
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Triton.Math.Softmax
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for SoftmaxReciprocal.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocalCorrect
open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

/- Only the current shared source is executable. Mathematical views below are
its dtype erasure; these local reductions expose that view to execution proofs. -/
attribute [local simp] ComputeKernel.eraseDType ComputeKernel.toAlgorithm?
  ComputeKernel.toAlgKernel ComputeKernel.fromKernelBody ComputeStmt.listToAlgorithm?
  ComputeStmt.toAlgorithm? ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm?
  Kernel.eraseDType Stmt.eraseDTypeList Stmt.eraseDType Op.eraseDType.eq_def
  MemAccess.eraseDType.eq_def MaskOpt.eraseDType.eq_def
  VeriTile.Triton.eraseDType NumericDType.eraseDType ComparableDType.eraseDType
  IntegralDType.eraseDType ComputeDType.eraseDType
attribute [local simp] stableSoftmaxKernel softmaxRecipKernel commonPrefix originalKernel reciprocalKernel

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- Executable real projection of per-lane division. -/
def divMathKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  (stableSoftmaxKernel xReg yReg blockSize).eraseDType

/-- Executable real projection of reciprocal multiplication. -/
def recipMathKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  (softmaxRecipKernel xReg yReg blockSize).eraseDType

theorem div_projection (xReg yReg : RegionName) (B : Nat) :
    (stableSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (divMathKernel xReg yReg B).toAlgKernel := by
  simp [divMathKernel, recipMathKernel]

theorem recip_projection (xReg yReg : RegionName) (B : Nat) :
    (softmaxRecipKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (recipMathKernel xReg yReg B).toAlgKernel := by
  simp [divMathKernel, recipMathKernel]

theorem div_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (divMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, divMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem div_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (divMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  try simp only [divMathKernel, recipMathKernel, stableSoftmaxKernel, softmaxRecipKernel, commonPrefix, originalKernel, reciprocalKernel, ComputeKernel.seq, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType, ComputeDType.eraseDType, Except.bind, bind]
  simp only [BlockState.pid_eq] at hx hy
  simp [divMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

theorem div_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (divMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hinj : Function.Injective (fun i : TileIndex [n+1] => s.pids 0 * (n+1) + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [divMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp only [hx]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem recip_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (recipMathKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, recipMathKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
theorem recip_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (recipMathKernel xReg yReg B).toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  try simp only [divMathKernel, recipMathKernel, stableSoftmaxKernel, softmaxRecipKernel, commonPrefix, originalKernel, reciprocalKernel, ComputeKernel.seq, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType, ComputeDType.eraseDType, Except.bind, bind]
  simp only [BlockState.pid_eq] at hx hy
  simp [recipMathKernel, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

theorem recip_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (recipMathKernel "x" "y" B).toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hinj : Function.Injective (fun i : TileIndex [n+1] => s.pids 0 * (n+1) + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [recipMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp only [hx, ← div_eq_mul_inv]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

end VeriTile.Bench.Examples.SoftmaxReciprocalCorrect
