import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Every original typed profile projects through the checked algorithm
interface before dtype erasure. This is not a floating-point equivalence. -/
theorem typed_projects (storage : Typed.Storage) (v : Bool) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.homogeneousKernel storage v GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgorithm? =
      .ok (Typed.homogeneousKernel storage v GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel := by
  cases storage <;> cases v <;> cases log <;> cases causal <;> cases fp16 <;> rfl

/-- Full-source execution after explicit mathematical dtype erasure agrees
with the previously checked real source. Branch continuation duplication may
change the syntax tree, so this is an interpreter equation, not an asserted
syntactic identity or a statement about native rounding. -/
theorem typed_erased_execution (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) (s : BlockState) :
    exec (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s =
    exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [Typed.homogeneousKernel, source, ksoftmax_backward_surface,
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

/-- The real output contract is linked to the complete typed source, with
dtype erasure explicit in the statement. All native precision claims remain
the separate typed FP equivalence and backend admission obligations. -/
theorem typed_erased_region_run (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool)
    (s : BlockState) (g o : Fin D → ℝ)
    (hg : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem GradOut (s.pids 0 * gm + s.pids 1 * gn + j.val) = g j)
    (ho : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem Out (s.pids 0 * om + s.pids 1 * on + j.val) = o j) :
    ∃ t, exec (Typed.homogeneousKernel storage Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s = some t ∧
      (∀ i : Fin D, i.val < K →
        t.readMem GradIn (s.pids 0 * bm + s.pids 1 * bn + i.val) =
          formula K (s.pids 1) log causal g o i) ∧
      (∀ r offset,
        (r ≠ GradIn ∨ ∀ i : Fin D, i.val < K → offset ≠ s.pids 0 * bm + s.pids 1 * bn + i.val) →
        t.mem r offset = s.mem r offset) := by
  rw [typed_erased_execution]
  exact source_region_run GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s g o hg ho

#axiomsClean typed_projects
#axiomsClean typed_erased_execution
#axiomsClean typed_erased_region_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
