import bench.tritonbench_g.matmul_triton2.Proofs.SourceFullExecution
import bench.tritonbench_g.matmul_triton2.Proofs.Grid
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Explicit validity of the source's runtime integer casts and arange casts.
Strides remain independent. This states the boundary needed by the pasted
uint64 body, rather than treating its casts as no-ops for arbitrary Naturals. -/
structure SourceBounds (p : Core.Parameters) : Prop where
  scalars : ∀ name n, (name,n) ∈ VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.scalarParameters p → n<2^64
  blockRows : p.blockRows≤2^64
  blockColumns : p.blockColumns≤2^64
  blockInner : p.blockInner≤2^64
  groupRows : p.groupRows<2^64
  launch : Geometry.count (Core.geometry p)≤2^64

theorem repaired_source_execution {α : Type} [Inhabited α] (rest : Algebra α)
    (P : Core.Primitives α) (p : Core.Parameters) (bounds : SourceBounds p)
    (initial : State (Word α)) (pid : initial.pids 0<Geometry.count (Core.geometry p)) :
    ∃ final, exec (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {})
      (VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.source p) initial=some final ∧
      final.mem=(Core.written (symbolicPrimitives α) p initial).mem := by
  exact source_execution_memory rest P p initial bounds.scalars
    (lt_of_lt_of_le pid bounds.launch) bounds.blockRows bounds.blockColumns bounds.blockInner bounds.groupRows

#axiomsClean repaired_source_execution
end
end VeriTile.Bench.Repairs.MatmulTriton2.SourceRuntime
