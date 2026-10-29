import bench.tritonbench_g.int8_quantization.Proofs.TypedScaled
namespace VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled
open VeriTile.Triton FP.Structural
noncomputable section
/-- The K profile has preScale=1. The multiply is retained: it is a real
software fp32 multiply, including its signed-zero/NaN behavior. -/
def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (i : TileIndex [p.blockRows,p.columns]) : FP.Value .fp32 :=
  FP.div cfg (FP.mul cfg (FP.Value.ofBits .fp32 0x3f800000)
    ((s.mem p.x (offset p s i)).read .real))
    ((s.mem p.suppliedScale (scaleOffset p s)).read .real)

theorem value_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (s : State (FP.Value .fp32)) (hscale : p.preScale = 1)
    (i : TileIndex [p.blockRows,p.columns]) :
    value (FP.Software.scalarFP32 rest cfg) p s i = roundedValue cfg p s i := by
  simp [value, scale, hscale, roundedValue, FP.Software.scalarFP32, FP.Software.fp32Context]
#axiomsClean value_software
end
end VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled
