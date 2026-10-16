import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTrace
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardIO

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

theorem full_typed_erased_projects (gradient saved destination : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgorithm? =
      .ok (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, full_typed_projects]
  simp

theorem full_typed_erased_flattenOk (gradient saved destination : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.FlattenOk := by
  simp only [ComputeKernel.eraseDType, full_typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases gradient <;> cases saved <;> cases destination
  all_goals
    dsimp only [Typed.kernel,
      Typed.kernel_fp32_fp32, Typed.kernel_fp32_fp64, Typed.kernel_fp32_fp16,
      Typed.kernel_fp32_bf16, Typed.kernel_fp32_f8e4, Typed.kernel_fp32_f8e5,
      Typed.kernel_fp64_fp32, Typed.kernel_fp64_fp64, Typed.kernel_fp64_fp16,
      Typed.kernel_fp64_bf16, Typed.kernel_fp64_f8e4, Typed.kernel_fp64_f8e5,
      Typed.kernel_fp16_fp32, Typed.kernel_fp16_fp64, Typed.kernel_fp16_fp16,
      Typed.kernel_fp16_bf16, Typed.kernel_fp16_f8e4, Typed.kernel_fp16_f8e5,
      Typed.kernel_bf16_fp32, Typed.kernel_bf16_fp64, Typed.kernel_bf16_fp16,
      Typed.kernel_bf16_bf16, Typed.kernel_bf16_f8e4, Typed.kernel_bf16_f8e5,
      Typed.kernel_f8e4_fp32, Typed.kernel_f8e4_fp64, Typed.kernel_f8e4_fp16,
      Typed.kernel_f8e4_bf16, Typed.kernel_f8e4_f8e4, Typed.kernel_f8e4_f8e5,
      Typed.kernel_f8e5_fp32, Typed.kernel_f8e5_fp64, Typed.kernel_f8e5_fp16,
      Typed.kernel_f8e5_bf16, Typed.kernel_f8e5_f8e4, Typed.kernel_f8e5_f8e5,
      ComputeKernel.surfaceBody,
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

theorem full_typed_erased_traceSafe (gradient saved destination : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hg : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * gm + s.pids 1 * gn + j.val < bounds GradOut)
    (ho : ∀ j : Fin D, readActive K (s.pids 1) causal j.val = Bool.true →
      s.pids 0 * om + s.pids 1 * on + j.val < bounds Out)
    (hb : ∀ j : Fin D, j.val < K →
      s.pids 0 * bm + s.pids 1 * bn + j.val < bounds GradIn) :
    Kernel.TraceSafe bounds
      (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s := by
  exact (full_typed_erased_trace_equiv gradient saved destination GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16 bounds s).mpr
      (source_traceSafe GradIn GradOut Out bm bn gm gn om on K D
        log causal fp16 bounds s hg ho hb)

/-- The IO object contains the actual typed source after explicit dtype
erasure. All three buffer dtypes vary independently, with the original source strides
and causal read mask retained. -/
def fullTypedErasedIO (gradient saved destination : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) : Masked2DKernelIO₂ where
  kernel := (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16).eraseDType
  projection := full_typed_erased_projects gradient saved destination GradIn GradOut Out bm bn gm gn om on K D log causal fp16
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
specification ksoftmax_backward_full_typed_real_correctness
    (gradient saved destination : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    fullTypedErasedIO gradient saved destination GradIn GradOut Out bm bn gm gn om on K D log causal fp16 ⊨
      fun _ row g o i => formula K row log causal g o i := by
  refine Masked2DKernelIO₂.Implements.intro _
    (full_typed_erased_flattenOk gradient saved destination GradIn GradOut Out bm bn gm gn om on K D log causal fp16) ?_ ?_
  · intro bounds s hg ho hb _
    exact full_typed_erased_traceSafe gradient saved destination GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 bounds s hg ho hb
  · intro s g o hg ho
    obtain ⟨t, hrun, hvalue, hframe⟩ := full_typed_erased_region_run gradient saved destination GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16 s g o hg ho
    exact ⟨t, hrun, hvalue, fun r offset hmiss _ => hframe r offset hmiss⟩

#axiomsClean full_typed_erased_projects
#axiomsClean full_typed_erased_flattenOk
#axiomsClean full_typed_erased_traceSafe
#axiomsClean ksoftmax_backward_full_typed_real_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
