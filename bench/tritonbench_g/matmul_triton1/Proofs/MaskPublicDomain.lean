import bench.tritonbench_g.matmul_triton1.Proofs.MaskFullExecution
import bench.tritonbench_g.matmul_triton1.Proofs.FiniteSource
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.MaskRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem scalar_parameters_small (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) :
    ∀ name n, (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters p → n < 2^64 := by
  intro name n member
  simp only [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with h | h | h | h | h | h | h | h | h
  all_goals cases h
  all_goals try simp only [canonical.inputRow,canonical.inputInner,canonical.weightInner,
    canonical.weightColumn,canonical.outputRow,canonical.outputColumn]
  all_goals have := ready.rows
  all_goals have := ready.columns
  all_goals have := ready.inner
  all_goals omega

theorem repaired_source_execution (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (initial : State (Word α)) (pid : initial.pids 0 < FiniteIndex.count p) :
    ∃ final, FP.Structural.exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P)
        (VeriTile.Bench.Repairs.MatmulTriton1.Kernel.maskHoisted p) initial = some final ∧
      final.mem = (Core.written (symbolicPrimitives α) p initial).mem := by
  apply source_execution_memory rest P p initial (scalar_parameters_small p canonical ready)
  · have := ready.launch
    omega
  · have := ready.blockRows.2
    omega
  · have := ready.blockColumns.2
    omega
  · have := ready.blockInner.2
    omega

#axiomsClean scalar_parameters_small
#axiomsClean repaired_source_execution
end
end VeriTile.Bench.Repairs.MatmulTriton1.MaskRuntime
