import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexLoop
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem write_related (p : ForwardLayout) (length : Nat) (a b : BlockState)
    (h : ForwardRelated p length a b) (r : RegionName) (offset : Nat) (v : ℝ) :
    ForwardRelated p length (a.writeMem r offset v) (b.writeMem r offset v) := by
  refine ⟨?_, h.pids, h.lengthReg, h.batchReg, h.dimReg, h.columns, h.mask, h.realState, h.imagState, h.realTransition, h.imagTransition⟩
  simp only [BlockState.writeMem, h.memory]

theorem store_related {ι : Type} (p : ForwardLayout) (length : Nat)
    (a b : BlockState) (h : ForwardRelated p length a b)
    (r : RegionName) (offset : ι → Nat) (v : ι → ℝ) (mask : ι → Bool) (lanes : List ι) :
    ForwardRelated p length
      (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) a)
      (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) b) := by
  induction lanes generalizing a b with
  | nil => exact h
  | cons i is ih =>
    simp only [List.foldl_cons]
    cases hm : mask i <;> simp only [hm, Bool.false_eq_true, ↓reduceIte]
    · exact ih a b h
    · exact ih _ _ (write_related p length a b h r (offset i) (v i))

theorem store_memory {ι : Type} (r : RegionName) (offset : ι → Nat) (v : ι → ℝ)
    (mask : ι → Bool) (lanes : List ι) (a b : BlockState) (h : a.mem = b.mem) :
    (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) a).mem =
    (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) b).mem := by
  induction lanes generalizing a b with
  | nil => exact h
  | cons i is ih =>
    simp only [List.foldl_cons]
    cases hm : mask i <;> simp only [hm, Bool.false_eq_true, ↓reduceIte]
    · exact ih a b h
    · apply ih
      simp only [BlockState.writeMem, h]

theorem forward_step_related (p : ForwardLayout) (length t : Nat) (a b : BlockState)
    (h : ForwardRelated p length a b) :
    ∃ va vb, stepStmts (forwardLoop p length)
        (a.setReg "t" .int [] (Tile.scalar (t : Int))) = some va ∧
      stepStmts (ComplexForwardSource.body p.x p.y p.batch p.dim p.B)
        (b.setReg "t" .nat [] (Tile.scalar t)) = some vb ∧
      ForwardRelated p length va vb := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hRa, hRb⟩, ⟨si, hIa, hIb⟩,
    ⟨lr, hLa, hLb⟩, ⟨li, hJa, hJb⟩⟩
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  rotate_left 2
  ·
    simp only [forwardLoop, forwardBody, Kernels.complexForwardTyped,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      ComputeDType.eraseDType, List.map_cons, List.map_nil,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
    simp only [List.getElem?_cons_zero, List.getElem?_cons_succ]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_masked_regs, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, bind, Option.bind, Option.map_some]
    simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hMa, hRa, hIa, hLa, hJa, BlockState.readMemValue, Region.cast_cast]
    simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Nat.mul_assoc,
      BlockState.readMemAs, BlockState.readMem, memory, BlockState.writeMemTyped]
    try simp only [Option.map_some, Option.bind, BlockState.foldl_writeMem_masked_regs,
      BlockState.foldl_writeMem_prop_masked_regs, BlockState.setReg,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hRa, hRb, hIa, hIb, hLa, hLb, hJa, hJb]
    rfl
  · simp only [ComplexForwardSource.body, Erasure.complexForward,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, bind, Bool.false_eq_true, ↓reduceIte, List.drop]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_masked_regs,
      BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
      bind, Option.bind, Option.map_some]
    simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hCb, hMb, hRb, hIb, hLb, hJb,
      BlockState.readMemValue, Region.cast_cast, BlockState.readMemAs, BlockState.readMem,
      BlockState.writeMemTyped]
    try simp only [Option.map_some, Option.bind, BlockState.foldl_writeMem_masked_regs,
      BlockState.foldl_writeMem_prop_masked_regs, BlockState.setReg,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hRa, hRb, hIa, hIb, hLa, hLb, hJa, hJb]
    rfl
  · simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
      Nat.mul_assoc, BlockState.writeMemAs_real, BlockState.foldl_writeMem_masked_regs]
    simp (config := {maxSteps := 2000000}) (disch := positivity) only [Int.toNat_add, Int.toNat_mul, Int.toNat_natCast, Int.reduceToNat]
    refine ⟨?_, ?_, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩, ?_, ?_, ⟨lr, ?_, ?_⟩, ⟨li, ?_, ?_⟩⟩
    · apply store_memory
      apply store_memory
      rfl
    · simpa only [BlockState.foldl_writeMem_prop_masked_pids] using pids
    all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hLa, hLb, hJa, hJb]
    all_goals exact ⟨_, rfl, rfl⟩

#axiomsClean write_related
#axiomsClean store_related
#axiomsClean store_memory
#axiomsClean forward_step_related
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
