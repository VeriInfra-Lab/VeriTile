import bench.tritonbench_g.layer_norm_liger.Kernels
import bench.tritonbench_g.token_attn_reduceV.Proofs.Mathematical_FPModel

namespace VeriTile.Bench.Repairs.LayerNormLiger
open VeriTile.Triton FP.Structural
abbrev SumPrimitive := VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.SumPrimitive
/-- One supplied reduction tree is shared by typed FP32 carriers and the
actual FP32-source real carrier. No result-level reduction law is assumed. -/
def withSum {A : Type} (M : Algebra A) (R : SumPrimitive A) : Algebra A :=
  {VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages.withSum M R with
    reduceSum := fun pr {_} ax keep xs => R.apply pr ax keep xs}
end VeriTile.Bench.Repairs.LayerNormLiger
