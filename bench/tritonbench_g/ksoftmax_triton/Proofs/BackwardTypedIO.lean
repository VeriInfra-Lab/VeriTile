import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasure

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

theorem typed_erased_projects (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgorithm? =
      .ok (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects]
  simp

theorem typed_erased_flattenOk (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.FlattenOk := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [Typed.homogeneousKernel, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem typed_erased_traceSafe (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hg : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * gm + s.pids 1 * gn + j.val < bounds GradOut)
    (ho : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * om + s.pids 1 * on + j.val < bounds Out)
    (hb : ∀ j : Fin D, j.val < K →
      s.pids 0 * bm + s.pids 1 * bn + j.val < bounds GradIn) :
    Kernel.TraceSafe bounds
      (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases log <;> cases causal <;> cases fp16
  all_goals
    simp [readActive] at hg ho
    dsimp only [Typed.homogeneousKernel, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    unfold Kernel.TraceSafe
    simp [Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
      Op.PointerAddressesSafeOn, Op.MemorySafe, BlockState.setReg, tile_elementwise]
    exact ⟨hg, ho, hb⟩

/-- The IO object contains the actual typed source after explicit dtype
erasure. Its buffers retain the distinct source strides and causal read mask. -/
def typedErasedIO (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) : Masked2DKernelIO₂ where
  kernel := (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16).eraseDType
  projection := typed_erased_projects storage GradIn GradOut Out bm bn gm gn om on K D log causal fp16
  in1 := GradOut
  in2 := Out
  out := GradIn
  B := D
  read1 := fun m n j => m * gm + n * gn + j.val
  read2 := fun m n j => m * om + n * on + j.val
  write := fun m n j => m * bm + n * bn + j.val
  mask := fun _ n j => readActive K n causal j.val = Bool.true
  writeMask := fun _ _ j => j.val < K

/-- Mathematical IO for the whole typed source. Dtype erasure is explicit;
machine rounding and native profile acceptance are separate obligations. -/
specification ksoftmax_backward_typed_real_correctness
    (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    typedErasedIO storage GradIn GradOut Out bm bn gm gn om on K D log causal fp16 ⊨
      fun _ row g o i => formula K row log causal g o i := by
  refine Masked2DKernelIO₂.Implements.intro _
    (typed_erased_flattenOk storage GradIn GradOut Out bm bn gm gn om on K D log causal fp16) ?_ ?_
  · intro bounds s hg ho hb _
    exact typed_erased_traceSafe storage GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 bounds s hg ho hb
  · intro s g o hg ho
    obtain ⟨t, hrun, hvalue, hframe⟩ := typed_erased_region_run storage GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 s g o hg ho
    exact ⟨t, hrun, hvalue, fun r offset hmiss _ => hframe r offset hmiss⟩

#axiomsClean typed_erased_projects
#axiomsClean typed_erased_flattenOk
#axiomsClean typed_erased_traceSafe
#axiomsClean ksoftmax_backward_typed_real_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
