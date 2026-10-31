import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceWhileExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def scatter (A : Algebra α) (p : Parameters) (pid : Nat) (acc : WholeSource.Acc α p) (initial : State α) : State α :=
  (TileShape.allIndices [p.blockRows,1]).foldl (fun state i =>
    if WholeSource.row p pid i.1.val<p.rows then
      state.write p.out (WholeSource.row p pid i.1.val*p.strideOut)
        (.mk .real (WholeSource.outputValues A p acc (i.1,())))
    else state) initial

def written (A : Algebra α) (p : Parameters) (initial : State α) (acc : WholeSource.Acc α p) : State α :=
  scatter A p (initial.pids 0) acc initial

def epilogueResult (A : Algebra α) (p : Parameters) (initial current : State α)
    (acc : WholeSource.Acc α p) : State α :=
  let s := current.setReg "acc" .real [p.blockRows] (WholeSource.outputValues A p acc)
  let s := s.setReg "C_ptrs" .ptr [p.blockRows,1]
    (fun i => (p.out,WholeSource.row p (initial.pids 0) i.1.val*p.strideOut))
  scatter A p (initial.pids 0) acc s

def epilogueExecution (A : Algebra α) (p : Parameters)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current) :
    {after : State α // run A (epilogue p) current=some after} := by
  refine ⟨epilogueResult A p initial current acc,?_⟩
  rcases h.context with ⟨hoffn,hoffm,hnMask,haptr,hbptr,hacc,hmem⟩
  have hstride := h.metadata "stride_cn" p.strideOut (by simp [scalarParameters])
  simp [epilogue,source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,FP.Structural.bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    epilogueResult,scatter,WholeSource.outputValues,hoffn,hnMask,hacc,hmem,h.pids,hstride,castId,
    Broadcast.leftIndex,Broadcast.rightIndex]
  congr 8 <;> (first | rfl | (funext i; simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]))

theorem epilogueMemory (A : Algebra α) (p : Parameters)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (initial current : State α) (off : Nat) (acc : WholeSource.Acc α p)
    (h : RuntimeInvariant p initial off acc current) :
    (epilogueExecution A p castId initial current off acc h).val.mem=(written A p initial acc).mem := by
  have memory : current.mem=initial.mem := h.context.2.2.2.2.2.2
  simp [epilogueExecution,epilogueResult,written,scatter,fold_writes_mem,memory,h.pids]

theorem source_execution_memory (A : Algebra α) (p : Parameters) (width : 0<p.blockColumns)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value) (initial : State α)
    (small : ∀ name n, (name,n) ∈ scalarParameters p → n<2^64)
    (pidSmall : initial.pids 0<2^64) (rowsSmall : p.blockRows≤2^64) (colsSmall : p.blockColumns≤2^64) :
    ∃ middle final, exec A (source p) initial=some final ∧
      AffineWhile.Trace A (AffineWhile.guard "m" "M") (loopBody p)
        (preludeExecution A p initial).val middle ∧
      final.mem=(written A p initial (WholeSource.accumulated A p (initial.pids 0)
        (viewMemory initial.mem) 0 (fun _ => A.literal (some .fp32) .real 0))).mem := by
  let start := preludeExecution A p initial
  have startReady := preludeInvariant A p initial small pidSmall rowsSmall colsSmall
  obtain ⟨last,middle,loop,trace,inv⟩ := whileExecution A p width castId loadId initial start.val 0
    (fun _ => A.literal (some .fp32) .real 0) startReady
  let finish := epilogueExecution A p castId initial middle last _ inv
  refine ⟨middle,finish.val,?_,trace,epilogueMemory A p castId initial middle last _ inv⟩
  change run A (source p).surfaceBody initial=some finish.val
  rw [source_decomposition,run_append,start.property]
  simp only [Option.bind_some,run,dynamicLoop A p initial start.val 0 _ startReady,loop]
  exact finish.property

#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean source_execution_memory
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
