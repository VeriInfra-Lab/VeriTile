import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardProfiles
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Every complete source in the independent three-storage family has a
checked algorithm projection. This says nothing about native admission. -/
theorem full_typed_projects (gradient saved destination : Typed.Storage) (v : Bool)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) :
    (Typed.kernel gradient saved destination v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).toAlgorithm? =
      .ok (Typed.kernel gradient saved destination v GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).toAlgKernel := by
  cases gradient <;> cases saved <;> cases destination <;> cases v <;> rfl

/-- Execution of the entire independently typed original after explicit
dtype erasure. The proof normalizes only static control flow; source loads,
casts, reductions and stores are still present in the theorem's kernel. -/
theorem full_typed_erased_execution (gradient saved destination : Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) (s : BlockState) :
    exec (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s =
    exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s := by
  simp only [ComputeKernel.eraseDType, full_typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases gradient <;> cases saved <;> cases destination <;>
    cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [Typed.kernel,
      Typed.kernel_fp32_fp32, Typed.kernel_fp32_fp64, Typed.kernel_fp32_fp16, Typed.kernel_fp32_bf16, Typed.kernel_fp32_f8e4, Typed.kernel_fp32_f8e5, Typed.kernel_fp64_fp32, Typed.kernel_fp64_fp64, Typed.kernel_fp64_fp16, Typed.kernel_fp64_bf16, Typed.kernel_fp64_f8e4, Typed.kernel_fp64_f8e5, Typed.kernel_fp16_fp32, Typed.kernel_fp16_fp64, Typed.kernel_fp16_fp16, Typed.kernel_fp16_bf16, Typed.kernel_fp16_f8e4, Typed.kernel_fp16_f8e5, Typed.kernel_bf16_fp32, Typed.kernel_bf16_fp64, Typed.kernel_bf16_fp16, Typed.kernel_bf16_bf16, Typed.kernel_bf16_f8e4, Typed.kernel_bf16_f8e5, Typed.kernel_f8e4_fp32, Typed.kernel_f8e4_fp64, Typed.kernel_f8e4_fp16, Typed.kernel_f8e4_bf16, Typed.kernel_f8e4_f8e4, Typed.kernel_f8e4_f8e5, Typed.kernel_f8e5_fp32, Typed.kernel_f8e5_fp64, Typed.kernel_f8e5_fp16, Typed.kernel_f8e5_bf16, Typed.kernel_f8e5_f8e4, Typed.kernel_f8e5_f8e5, source, ksoftmax_backward_surface,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Except.bind, Except.pure, Bind.bind, Pure.pure,
      Kernel.eraseDType, ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      FloatDType.toTileDType, Region.cast]
    simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
    simp only [exec, StaticControl.cons_bind, stepStmts.nil,
      StaticControl.if_true, StaticControl.if_false, StaticControl.assign_real_cast,
      StaticControl.ifElse_true, StaticControl.ifElse_false,
      Option.bind_assoc, Option.bind_some, Option.bind_none, Option.bind_fun_some]
    rfl

/-- Independent finite-sum output and complete memory frame for the typed
source's mathematical erasure, for every storage triple and flag profile. -/
theorem full_typed_erased_region_run (gradient saved destination : Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) (s : BlockState) (g o : Fin D → ℝ)
    (hg : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem GradOut (s.pids 0 * gm + s.pids 1 * gn + j.val) = g j)
    (ho : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem Out (s.pids 0 * om + s.pids 1 * on + j.val) = o j) :
    ∃ t, exec (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s = some t ∧
      (∀ i : Fin D, i.val < K →
        t.readMem GradIn (s.pids 0 * bm + s.pids 1 * bn + i.val) =
          formula K (s.pids 1) log causal g o i) ∧
      (∀ r offset,
        (r ≠ GradIn ∨ ∀ i : Fin D, i.val < K → offset ≠ s.pids 0 * bm + s.pids 1 * bn + i.val) →
        t.mem r offset = s.mem r offset) := by
  rw [full_typed_erased_execution]
  exact source_region_run GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s g o hg ho

#axiomsClean full_typed_projects
#axiomsClean full_typed_erased_execution
#axiomsClean full_typed_erased_region_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
