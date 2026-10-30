import bench.repairs.tritonbench_g.dequantize_matmul.PairIO
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.DequantizeMatmul.PairEquivalence
open VeriTile Triton FP.Structural KernelSource KernelModels
open scoped VeriTile.Spec
noncomputable section

/-- The existing exact integer mask-product proof already covers both safe
full grids, every output cell and both exterior frames on signed-byte inputs. -/
theorem equivalent (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p) :
    PairIO.original multiply rest p ≡[[]] PairIO.indexCommuted multiply rest p := by
  apply ProgramIO.Implementation.of_equivalentOn
  intro values _
  exact PublicIO.equivalent (nativeAlgebra multiply rest) p hK hN sep values trivial

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.DequantizeMatmul.PairEquivalence
