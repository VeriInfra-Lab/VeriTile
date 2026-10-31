import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateFull
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceSafety
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem body_safe (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns) (bounds : RegionName → Nat)
    (initial current : State Payload) (off : Nat) (acc : WholeSource.Acc Payload p)
    (h : RuntimeInvariant p initial off acc current)
    (matrix : ∀ i : Fin p.blockRows, ∀ j : Fin p.blockColumns,
      WholeSource.row p (initial.pids 0) i.val<p.rows → off+j.val<p.columns →
        WholeSource.aOffset p (initial.pids 0) off i.val j.val<bounds p.a)
    (vector : ∀ j : Fin p.blockColumns, off+j.val<p.columns → WholeSource.bOffset p off j.val<bounds p.b) :
    Safety.traceSafeList (model native p g) bounds (loopBody p) current := by
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hM := h.metadata "M" p.columns (by simp [scalarParameters])
  have hAM := h.metadata "stride_am" p.strideColumn (by simp [scalarParameters])
  have hBM := h.metadata "stride_bm" p.strideVector (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  simp [loopBody,loopStatement,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.fmaBody,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,evalOffsets,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    model,VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra,WholeNative.algebra,
    hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem,hM,hAM,hBM,counter,State.setReg]
  constructor
  · intro i j activeRow activeColumn
    exact matrix i j activeRow activeColumn
  · intro j activeColumn
    exact vector j activeColumn

#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
