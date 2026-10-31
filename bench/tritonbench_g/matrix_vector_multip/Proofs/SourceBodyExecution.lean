import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceStorageView
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def bodyResult (A : Algebra α) (p : Parameters) (initial current : State α)
    (off : Nat) (acc : WholeSource.Acc α p) : State α :=
  let s := current.setReg "m_mask" .bool [1,p.blockColumns]
    (fun i => decide (off+i.2.1.val<p.columns))
  let s := s.setReg "a" .real [p.blockRows,p.blockColumns]
    (WholeSource.aChunk A p (initial.pids 0) off (viewMemory initial.mem))
  let s := s.setReg "b" .real [1,p.blockColumns]
    (WholeSource.bChunk A p off (viewMemory initial.mem))
  let s := s.setReg "acc" .real [p.blockRows,p.blockColumns]
    (WholeSource.nextAcc A p (initial.pids 0) off (viewMemory initial.mem) acc)
  let s := s.setReg "A_ptrs" .ptr [p.blockRows,p.blockColumns]
    (fun i => (p.a,WholeSource.aOffset p (initial.pids 0) (off+p.blockColumns) i.1.val i.2.1.val))
  let s := s.setReg "B_ptrs" .ptr [1,p.blockColumns]
    (fun i => (p.b,WholeSource.bOffset p (off+p.blockColumns) i.2.1.val))
  s.setReg "m" .nat [] (fun _ => off+p.blockColumns)

def bodyExecution (A : Algebra α) (p : Parameters)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current) :
    {after : State α // run A (loopBody p) current=some after} := by
  refine ⟨bodyResult A p initial current off acc,?_⟩
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hN := h.metadata "N" p.rows (by simp [scalarParameters])
  have hM := h.metadata "M" p.columns (by simp [scalarParameters])
  have hAM := h.metadata "stride_am" p.strideColumn (by simp [scalarParameters])
  have hBM := h.metadata "stride_bm" p.strideVector (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  simp [loopBody,loopStatement,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,castId,loadId,
    bodyResult,WholeSource.nextAcc,WholeSource.aChunk,WholeSource.bChunk,WholeSource.load,
    viewMemory,storageView_read,hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem,hN,hM,hAM,hBM,counter,
    Broadcast.leftIndex,Broadcast.rightIndex,WholeSource.aOffset,WholeSource.bOffset,
    Nat.add_mul,Nat.add_assoc,show (0.0 : ℝ)=0 by norm_num]
  congr 12 <;> (first | rfl | (funext i; simp [WholeSource.nextAcc,WholeSource.aChunk,
    WholeSource.bChunk,WholeSource.load,viewMemory,storageView_read,castId,loadId,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,WholeSource.aOffset,
    WholeSource.bOffset,Nat.add_mul,Nat.add_assoc]))

#axiomsClean bodyExecution
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
