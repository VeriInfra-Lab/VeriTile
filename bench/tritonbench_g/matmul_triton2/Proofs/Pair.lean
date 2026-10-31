import bench.tritonbench_g.matmul_triton2.Proofs.SourceIOCorrect
import bench.tritonbench_g.matmul_triton2.Proofs.MaskIOCorrect
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

theorem equivalent {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (scope : SourceRuntime.SourceBounds p)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (IO.matmul rest P p).Equivalent (variant rest P p) := by
  exact ProgramIO.EquivalentOn.of_implementsOn
    (SourceIOProof.precision rest P p scope geometry ha hb)
    (MaskIOProof.precision rest P p scope geometry ha hb)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Pair
