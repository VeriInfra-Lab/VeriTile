import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.GridExecution
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.Targets

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole

/-- Source geometry and regions only. Device and Tensor admission are absent. -/
structure Geometry where
  (A B C : RegionName)
  (rows inner columns blockRows blockColumns blockInner : Nat)
  positiveRowsBlock : 0<blockRows
  positiveColumnsBlock : 0<blockColumns
  positiveInnerBlock : 0<blockInner
  schedule : VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule

def Geometry.request (g : Geometry) : Request :=
  ⟨g.A,g.B,g.C,g.rows,g.inner,g.columns,g.blockRows,g.blockColumns,g.blockInner,
    g.positiveRowsBlock,g.positiveColumnsBlock,g.positiveInnerBlock,g.schedule,""⟩

/-- Any former request supplies exactly these raw source arguments. Its device
annotation is irrelevant to the quotation, grid and target. -/
def Geometry.ofRequest (r : Request) : Geometry :=
  ⟨r.A,r.B,r.C,r.rows,r.inner,r.columns,r.blockRows,r.blockColumns,r.blockInner,
    r.positiveRowsBlock,r.positiveColumnsBlock,r.positiveInnerBlock,r.schedule⟩

theorem grid_geometry {α : Type} [Inhabited α] (M : Algebra α) (graph : Graph)
    (r : Request) (initial : State α) :
    grid M graph (Geometry.ofRequest r).request initial=grid M graph r initial := by
  unfold grid
  change runPrograms M graph (Geometry.ofRequest r).request (List.range (programs r)) initial =
    runPrograms M graph r (List.range (programs r)) initial
  generalize List.range (programs r)=ids
  induction ids generalizing initial with
  | nil => rfl
  | cons pid ids ih =>
    change (exec M (quotation r.schedule graph r.parameters)
      {initial with pids:=fun ax=>if ax=0 then pid else 0}).bind
        (runPrograms M graph (Geometry.ofRequest r).request ids) =
      (exec M (quotation r.schedule graph r.parameters)
      {initial with pids:=fun ax=>if ax=0 then pid else 0}).bind (runPrograms M graph r ids)
    rw [show runPrograms M graph (Geometry.ofRequest r).request ids=runPrograms M graph r ids from funext ih]

#axiomsClean grid_geometry
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
