import bench.tritonbench_g.matmul_triton2.Proofs.SourceLoopExecution
import bench.tritonbench_g.matmul_triton2.Proofs.FPComplete
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def epilogueExecution (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial current : State (Word α)) (h : RuntimeInvariant p initial (Core.steps p) current) :
    {after : State (Word α) // run (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {}) (epilogue p) current=some after} := by
  have hm := h.metadata "M" p.rows (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hn := h.metadata "N" p.columns (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hscm := h.metadata "stride_cm" p.outputRow (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  have hscn := h.metadata "stride_cn" p.outputColumn (by simp [VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters])
  refine ⟨?_,?_⟩
  swap
  simp [epilogue,VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.source,VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.arguments,
    VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.body,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,
    ScalarArgument.integer,ScalarArgument.statement,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    hscm,hscn,hm,hn,h.offm,h.offn,h.acc,VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra]
  rfl

theorem epilogueMemory (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial current : State (Word α)) (h : RuntimeInvariant p initial (Core.steps p) current) :
    (epilogueExecution rest P p initial current h).val.mem =
      (Core.written (symbolicPrimitives α) p initial).mem := by
  simp [epilogueExecution,Core.written,masked_scatter_memory,h.memory,Core.active,Core.cOffset,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem source_execution_memory (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial : State (Word α))
    (small : ∀ name n, (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters p → n < 2^64)
    (pidSmall : initial.pids 0 < 2^64)
    (bmSmall : p.blockRows ≤ 2^64) (bnSmall : p.blockColumns ≤ 2^64)
    (bkSmall : p.blockInner ≤ 2^64) (groupSmall : p.groupRows < 2^64) :
    ∃ final, FP.Structural.exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {})
        (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.source p) initial = some final ∧
      final.mem = (Core.written (symbolicPrimitives α) p initial).mem := by
  let start := preludeExecution rest P p initial
  have startInvariant := preludeInvariant rest P p initial small pidSmall bmSmall bnSmall bkSmall groupSmall
  obtain ⟨middle,hm,hi⟩ := loopExecution rest P p initial start.val 0 (by omega) startInvariant
  let finish := epilogueExecution rest P p initial middle hi
  refine ⟨finish.val,?_,epilogueMemory rest P p initial middle hi⟩
  change run _ (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.source p).surfaceBody initial = some finish.val
  rw [source_decomposition,run_append,start.property]
  simp only [Option.bind_some,run,dynamicLoop rest P p initial start.val 0 startInvariant,hm]
  exact finish.property

#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean source_execution_memory
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
