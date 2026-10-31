import bench.tritonbench_g.matmul_triton1.Proofs.SourceFiniteOperands
import bench.tritonbench_g.matmul_triton1.Proofs.SourceGrid

namespace VeriTile.Bench.Repairs.MatmulTriton1.RawKernel
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem execute {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (initial : State (DotProfile.Value α)) :
    ∃ final, SourceRuntime.grid rest P p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (Core.geometry p))
        (fun i : Core.LogicalIndex p => Cell.mk .real
          (Core.logical (SourceRuntime.symbolicPrimitives α) p initial i.1.val i.2.val))
        (fun _ => True) initial final :=
  SourceRuntime.grid_correct rest P p canonical ready geometry ha hb initial

#axiomsClean execute
end
end VeriTile.Bench.Repairs.MatmulTriton1.RawKernel
