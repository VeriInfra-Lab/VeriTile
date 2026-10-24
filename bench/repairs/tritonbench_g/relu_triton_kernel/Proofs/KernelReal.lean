import bench.repairs.tritonbench_g.relu_triton_kernel.Kernels
import VeriTile.Examples.Common
import VeriTile.Triton.Memory.Flatten
import VeriTile.Meta.StatementAudit

/-! Mathematical kernel proof after removing the defective pid-zero gate.
The readback/frame argument reuses the original masked scatter infrastructure;
it covers every active lane of every program, including tail and empty tiles. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.KernelReal
open VeriTile.Triton VeriTile.Examples
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000

/-- Independent mathematical ReLU. -/
def formula (xs : Nat → ℝ) (i : Nat) : ℝ := max 0 (xs i)

theorem projection (N : Nat) :
    (originalKernel N).toAlgorithm? = .ok (originalKernel N).toAlgKernel := by rfl

theorem result (N : Nat) (s : BlockState) (xs : Fin blockSize → ℝ)
    (hx : ∀ j : Fin blockSize, s.pid * blockSize + j.val < N →
      s.readMem "x" (s.pid * blockSize + j.val) = xs j) :
    ∀ i : Fin blockSize,
      observeAt (exec (originalKernel N) s) "out" blockSize s.pid i =
        some (if s.pid * blockSize + i.val < N then max 0 (xs i)
          else s.readMem "out" (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, originalKernel, kernel, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, ComparableDType.ge, Region.cast, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType]
  rw [BlockState.scatter_readback_prop_masked_nd _
    (fun j : TileIndex [blockSize] => s.pids 0 * blockSize + j.1.val) _
    (fun j : TileIndex [blockSize] => s.pids 0 * blockSize + j.1.val < N)
    (fun a b h => by
      apply Prod.ext
      · apply Fin.ext
        exact Nat.add_left_cancel h
      · exact Subsingleton.elim _ _) (i, PUnit.unit)]
  by_cases hi : s.pid * blockSize + i.val < N
  · simp [hi, hx i hi]
    split_ifs with hge
    · have h : (0 : ℝ) ≤ xs i := WithBot.coe_le_coe.mp hge
      exact (max_eq_right h).symm
    · have h : ¬ ((0 : ℝ) ≤ xs i) := fun h => hge (WithBot.coe_le_coe.mpr h)
      rw [max_eq_left (le_of_not_ge h)]
      norm_num
  · simp [hi]

theorem frame (N : Nat) (s t : BlockState)
    (he : exec (originalKernel N).toAlgKernel s = some t)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin blockSize, s.pid * blockSize + i.val < N →
      ¬("out" = r ∧ s.pid * blockSize + i.val = o)) :
    t.mem r o = s.mem r o := by
  simp [exec, originalKernel, kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    Region.cast, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType] at he
  subst he
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 hmk hc

theorem region_run (N : Nat) (s : BlockState) (xs : Fin blockSize → ℝ)
    (hx : ∀ j : Fin blockSize, s.pid * blockSize + j.val < N →
      s.readMem "x" (s.pid * blockSize + j.val) = xs j) :
    ∃ t, exec (originalKernel N).toAlgKernel s = some t ∧
      (∀ j : Fin blockSize, s.pid * blockSize + j.val < N →
        t.readMem "out" (s.pid * blockSize + j.val) = max 0 (xs j)) ∧
      (∀ (r : RegionName) (o : Nat), (r ≠ "out" ∨ ∀ j : Fin blockSize,
        s.pid * blockSize + j.val < N → o ≠ s.pid * blockSize + j.val) →
        t.mem r o = s.mem r o) := by
  have hobs := result N s xs hx
  change ∀ i, observeAt (exec (originalKernel N).toAlgKernel s) _ _ _ i = _ at hobs
  cases he : exec (originalKernel N).toAlgKernel s with
  | none =>
    have := hobs ⟨0, by decide⟩
    rw [he] at this
    simp [observeAt] at this
  | some t =>
    refine ⟨t, rfl, ?_, ?_⟩
    · intro j hj
      have h := hobs j
      rw [he] at h
      simpa [observeAt, hj] using h
    · intro r o outside
      apply frame N s t he r o
      intro j hj ⟨hr, ho⟩
      rcases outside with hr' | ho'
      · exact hr' hr.symm
      · exact ho' j hj ho.symm

theorem flattenOk (N : Nat) : ((originalKernel N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [originalKernel, kernel, ComputeKernel.toAlgKernel, Region.cast, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

theorem traceSafe (N : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin blockSize, s.pid * blockSize + j.val < N →
      s.pid * blockSize + j.val < bounds "x")
    (hout : ∀ j : Fin blockSize, s.pid * blockSize + j.val < N →
      s.pid * blockSize + j.val < bounds "out") :
    Kernel.TraceSafe bounds (originalKernel N).toAlgKernel s := by
  unfold Kernel.TraceSafe
  simp [originalKernel, kernel, ComputeKernel.toAlgKernel, Region.cast, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

#specNonCircular formula avoiding [kernel, originalKernel, fusedKernel]
#axiomsClean region_run
#axiomsClean traceSafe
end
end VeriTile.Bench.Repairs.ReluTritonKernel.KernelReal
