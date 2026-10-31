import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceFullExecution
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem pureList_safe (A : Algebra β) [Inhabited β] (bounds : RegionName → Nat)
    (code : List ComputeStmt) :
    (∀ statement ∈ code, ∀ s, Safety.stepSafe A bounds statement s) →
      ∀ s, Safety.traceSafeList A bounds code s := by
  induction code with
  | nil => intro _ s; simp
  | cons statement rest ih =>
    intro allSafe s
    apply Safety.traceSafeList_cons_intro (allSafe statement (by simp) s)
    intro next _
    exact ih (fun stmt member => allSafe stmt (List.mem_cons.mpr (Or.inr member))) next

theorem prelude_safe (A : Algebra α) (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds (prelude p) s := by
  apply pureList_safe
  simp [prelude,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask]

theorem body_safe (A : Algebra α) (p : Parameters) (bounds : RegionName → Nat)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current)
    (matrix : ∀ i : Fin p.blockRows, ∀ j : Fin p.blockColumns,
      WholeSource.row p (initial.pids 0) i.val<p.rows → off+j.val<p.columns →
        WholeSource.aOffset p (initial.pids 0) off i.val j.val<bounds p.a)
    (vector : ∀ j : Fin p.blockColumns, off+j.val<p.columns → WholeSource.bOffset p off j.val<bounds p.b) :
    Safety.traceSafeList A bounds (loopBody p) current := by
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hM := h.metadata "M" p.columns (by simp [scalarParameters])
  have hAM := h.metadata "stride_am" p.strideColumn (by simp [scalarParameters])
  have hBM := h.metadata "stride_bm" p.strideVector (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  simp [loopBody,loopStatement,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,evalOffsets,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem,hM,hAM,hBM,counter,State.setReg]
  constructor
  · intro i j activeRow activeColumn
    exact matrix i j activeRow activeColumn
  · intro j activeColumn
    exact vector j activeColumn

theorem epilogue_safe (A : Algebra α) (p : Parameters) (bounds : RegionName → Nat)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current)
    (output : ∀ i : Fin p.blockRows, WholeSource.row p (initial.pids 0) i.val<p.rows →
      WholeSource.row p (initial.pids 0) i.val*p.strideOut<bounds p.out) :
    Safety.traceSafeList A bounds (epilogue p) current := by
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hstride := h.metadata "stride_cn" p.strideOut (by simp [scalarParameters])
  simp [epilogue,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,evalOffsets,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    hoffn,hnMask,hacc,hmem,hstride,State.setReg]
  intro i active
  simpa using output i active

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
