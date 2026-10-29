import bench.repairs.tritonbench_g.dequantize_matmul.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Two existing complete typed launch schedules on the original signed-byte
input domain and fixed public IO. All source precision and observations remain. -/
namespace VeriTile.Bench.Repairs.DequantizeMatmul.PairIO
open VeriTile Triton FP.Structural KernelSource KernelModels
noncomputable section

def original (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters) :=
  ProgramIO.Implementation.original (PublicIO.sourceIO (nativeAlgebra multiply rest) p Bool.false)
    PublicIO.SignedBytes

def indexCommuted (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters) :=
  ProgramIO.Implementation.ofVariant (PublicIO.variant (nativeAlgebra multiply rest) p)
    PublicIO.SignedBytes

end
end VeriTile.Bench.Repairs.DequantizeMatmul.PairIO
