import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceBodyExecution
import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_WholeNative
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

noncomputable abbrev source := Kernels.fmaFromParameters
def prelude (p : Parameters) : List ComputeStmt := (source p).surfaceBody.take 20
def loopStatement (p : Parameters) : ComputeStmt :=
  ((source p).surfaceBody[20]?).getD (.ifThen (.constBool Bool.false) [])
def loopBody (p : Parameters) : List ComputeStmt := match loopStatement p with
  | .forRangeDyn _ _ _ _ code => code
  | _ => []
def epilogue (p : Parameters) : List ComputeStmt := (source p).surfaceBody.drop 21

theorem source_decomposition (p : Parameters) :
    (source p).surfaceBody=prelude p++loopStatement p::epilogue p := by rfl

theorem prelude_eq (p : Parameters) : prelude p=VeriTile.Bench.Repairs.MatrixVectorMultip.Source.prelude p := by rfl
theorem epilogue_eq (p : Parameters) : epilogue p=VeriTile.Bench.Repairs.MatrixVectorMultip.Source.epilogue p := by rfl

def model (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns) : Algebra Payload :=
  VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra native g

def bodyExecution (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (initial current : State Payload) (off : Nat) (acc : WholeSource.Acc Payload p)
    (h : RuntimeInvariant p initial off acc current) :
    {after : State Payload // run (model native p g) (loopBody p) current=some after} := by
  refine ⟨bodyResult (WholeNative.algebra native g) p initial current off acc,?_⟩
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hN := h.metadata "N" p.rows (by simp [scalarParameters])
  have hM := h.metadata "M" p.columns (by simp [scalarParameters])
  have hAM := h.metadata "stride_am" p.strideColumn (by simp [scalarParameters])
  have hBM := h.metadata "stride_bm" p.strideVector (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  simp [loopBody,loopStatement,source,Kernels.fmaFromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.fmaBody,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    model,VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra,WholeNative.algebra,
    bodyResult,WholeSource.nextAcc,WholeSource.aChunk,WholeSource.bChunk,WholeSource.load,
    viewMemory,storageView_read,hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem,hN,hM,hAM,hBM,counter,
    Broadcast.leftIndex,Broadcast.rightIndex,WholeSource.aOffset,WholeSource.bOffset,
    Nat.add_mul,Nat.add_assoc,show (0.0 : ℝ)=0 by norm_num]
  congr 12 <;> (first | rfl | (funext i; simp [WholeSource.nextAcc,WholeSource.aChunk,
    WholeSource.bChunk,WholeSource.load,viewMemory,storageView_read,
    model,VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra,WholeNative.algebra,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,WholeSource.aOffset,
    WholeSource.bOffset,Nat.add_mul,Nat.add_assoc]))

#axiomsClean source_decomposition
#axiomsClean prelude_eq
#axiomsClean epilogue_eq
#axiomsClean bodyExecution
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
