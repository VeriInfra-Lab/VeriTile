import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Core
noncomputable section
variable {α : Type} [Inhabited α]

def variant (P : Source.Arithmetic α α) (p : Parameters) :
    ProgramIO.Variant (PublicData.io P p) where
  schedule := (Pair.candidateIO P p).schedule
  observe := (PublicData.io P p).observe

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PairIO
