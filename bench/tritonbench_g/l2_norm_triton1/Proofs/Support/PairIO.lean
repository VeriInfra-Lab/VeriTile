import bench.tritonbench_g.l2_norm_triton1.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.L2NormTriton1.PairIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete original row grid, including the actual runtime epsilon argument. -/
def original {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.original (GridIO.gridIO A rows N B eps) (fun _ => True)

/-- Existing candidate: both row-base products change from `row*N` to `N*row`.
The complete candidate invocation list and its observer are retained as data. -/
def commuted {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :=
  ProgramIO.Implementation.ofVariant (GridIO.candidate A rows N B eps) (fun _ => True)

end
end VeriTile.Bench.Repairs.L2NormTriton1.PairIO
