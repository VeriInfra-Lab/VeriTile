import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONativeGraph
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Candidate
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
open MatmulDequantizeInt4.FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section

/-- Native graph projection of the ordinary complete compiled quotation.
The remaining phases are exactly the already established shared native graph. -/
def preludeKernel (p : Parameters) : ComputeKernel :=
  .mk [p.scale,p.zero] [] ((groupedCompiledSource p).surfaceBody.take 24)

def programJobs (actual : Native.Instructions) (op : Opcode) (p : Parameters) (pid : Nat) :
    List (Invocation Word) :=
  ([IOExecution.launch (preludeKernel p) (IONativeGraph.model actual op) pid] ++
    IONativeGraph.loopJobs actual op p 0 (steps p)) ++
      [IONativeGraph.held (IONativeGraph.epilogueKernel p) (IONativeGraph.model actual op)]

def jobs (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :
    List (Invocation Word) :=
  (List.range (KernelSource.count c.call i)).flatMap
    (programJobs actual (tunings i).opcode (params c.call i))

def program (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :=
  IOData.program Word c (IONativeGraph.jobs actual c i)

def variant (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :
    ProgramIO.Variant (program actual c i) where
  schedule := fun _ => jobs actual c i
  observe := fun _ _ => id

/-- The same changed prelude belongs to the complete original-source candidate
and its complete compiled candidate; every later phase is unchanged. -/
theorem prelude_binding (p : Parameters) :
    (groupedSource p).surfaceBody.take 24=(groupedCompiledSource p).surfaceBody.take 24 := rfl

theorem source_tail (p : Parameters) :
    (groupedSource p).surfaceBody.drop 7=(FPTyped.source p).surfaceBody.drop 7 := rfl

theorem compiled_tail (p : Parameters) :
    (groupedCompiledSource p).surfaceBody.drop 7=(FPCompiled.source p).surfaceBody.drop 7 := rfl
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Candidate
