import bench.tritonbench_g.matmul_triton2.Proofs.SourceDomain
import bench.tritonbench_g.matmul_triton2.Proofs.MaskFullExecution
import bench.tritonbench_g.matmul_triton2.Proofs.Grid
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Explicit validity of the source's runtime integer casts and arange casts.
Strides remain independent. This states the boundary needed by the pasted
uint64 body, rather than treating its casts as no-ops for arbitrary Naturals. -/
abbrev SourceBounds := VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime.SourceBounds

theorem repaired_source_execution {α : Type} [Inhabited α] (rest : Algebra α)
    (P : Core.Primitives α) (p : Core.Parameters) (bounds : SourceBounds p)
    (initial : State (Word α)) (pid : initial.pids 0<Geometry.count (Core.geometry p)) :
    ∃ final, exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {})
      (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.maskHoisted p) initial=some final ∧
      final.mem=(Core.written (symbolicPrimitives α) p initial).mem := by
  exact source_execution_memory rest P p initial bounds.scalars
    (lt_of_lt_of_le pid bounds.launch) bounds.blockRows bounds.blockColumns bounds.blockInner bounds.groupRows

#axiomsClean repaired_source_execution
end
end VeriTile.Bench.Repairs.MatmulTriton2.MaskRuntime
