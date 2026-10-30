import bench.tritonbench_g.fp4_to_bf16.Sources.Mathematical_Kernels
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast
import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FP4Bits

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def embed (x : FP.Value .bf16) : Word := ⟨x.bits.zeroExtend 32⟩
def backend (rest : Algebra Word) : Algebra Word :=
  { rest with fp32Bits := fun bits => ⟨bits.bits⟩
              cast := fun precision src dst x =>
                match src,dst with
                | .real,.bf16 | .fp32,.bf16 => embed (Bits.narrow x)
                | src,dst => rest.cast precision src dst x }
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
