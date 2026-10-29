import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import VeriTile.Triton.Memory.Denotation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceFlat
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- Flattening depends on the final memory, not the temporary registers. -/
theorem flat_memory (A : FlatAlloc) (a b : BlockState) (h : a.mem = b.mem) :
    (A.flattenState a).mem = (A.flattenState b).mem := by
  funext r o
  simp only [FlatAlloc.flattenState, FlatAlloc.readFlat, h]

/-- Transfer a known flat result through two checked regional executions.
All safety and supported-fragment obligations are explicit and proved below
for the actual source; no historical register equality is required. -/
theorem transfer (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (actual reference : Kernel) (u a b old : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (ha : exec actual u = some a) (hb : exec reference u = some b)
    (hm : a.mem = b.mem)
    (sa : actual.TraceSafe A.extent u) (sb : reference.TraceSafe A.extent u)
    (oa : actual.FlattenOk) (ob : reference.FlattenOk)
    (he : exec (A.flattenKernel reference) (A.flattenState u) = some old) :
    ∃ v, exec (A.flattenKernel actual) (A.flattenState u) = some v ∧ v.mem = old.mem := by
  rw [A.exec_flatten hd hc reference u sb ob hu, hb, Option.map_some] at he
  have eq := Option.some.inj he
  refine ⟨A.flattenState a, ?_, ?_⟩
  · rw [A.exec_flatten hd hc actual u sa oa hu, ha, Option.map_some]
  · rw [← eq]
    exact flat_memory A a b hm

theorem reference_forward_flatten (p : ForwardLayout) (length : Nat) :
    (RealReference.diag_ssm_forward_kernel p.s p.x p.l p.y length p.batch p.dim p.B).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  rw [RealReference.diag_ssm_forward_kernel_toAlg_body]
  simp [RealReference.diagSsmForwardProjectedBody, RealReference.diagSsmForwardPreLoop,
    RealReference.diagSsmForwardLoopBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem reference_backward_flatten (p : BackwardLayout) (length : Nat) :
    (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  rw [RealReference.diag_ssm_backward_kernel_toAlg_body]
  simp [RealReference.diagSsmBackwardProjectedBody, RealReference.diagSsmBackwardPreLoop,
    RealReference.diagSsmBackwardLoopBody, RealReference.diagSsmBackwardPostLoop,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem forward_flatten (p : ForwardLayout) (length : Nat) :
    (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Kernels.forwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem backward_flatten (p : BackwardLayout) (length : Nat) :
    (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Kernels.backwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem complexForward_flatten (p : ForwardLayout) (length : Nat) :
    (Kernels.complexForwardTyped p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Kernels.complexForwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem complexBackward_flatten (p : BackwardLayout) (length : Nat) :
    (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Kernels.complexBackwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

#axiomsClean transfer
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceFlat
