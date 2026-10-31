import bench.tritonbench_g.matmul_triton1.Proofs.SourceIOCorrect
import bench.tritonbench_g.matmul_triton1.Proofs.MaskIOCorrect
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulTriton1.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem equivalent {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (MatmulIO.io rest P p).Equivalent (variant rest P p) := by
  exact ProgramIO.EquivalentOn.of_implementsOn
    (SourceIOProof.precision rest P p canonical ready storage geometry ha hb)
    (MaskIOProof.precision rest P p canonical ready storage geometry ha hb)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulTriton1.Pair
