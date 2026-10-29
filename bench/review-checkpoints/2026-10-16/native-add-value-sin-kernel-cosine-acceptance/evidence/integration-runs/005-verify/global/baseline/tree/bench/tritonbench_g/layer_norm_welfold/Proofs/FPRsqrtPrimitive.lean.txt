import bench.tritonbench_g.layer_norm_welfold.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (word sequentialFP32 sequentialSum)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 15000

/-- fp32 encoding of the source literal 1e-05, rounded to nearest-even. -/
def epsilon32 : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3727c5ac

theorem epsilon32_encoding :
    FP.round .fp32 .ieeeValue (1 / 100000 : Rat) = epsilon32 := by decide +kernel

/-- Concrete scalar arithmetic, integer conversion and sequential sum. The
remaining rsqrt primitive is explicit; no reciprocal-square-root law is assumed. -/
def fp32WithRsqrt (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) : Algebra (FP.Value .fp32) :=
  { sequentialFP32 rest cfg with
    fromNat := fun p n => if p = some .fp32 then FP.round .fp32 cfg (n : Rat) else rest.fromNat p n
    literal := fun p d x => if ((p = none ∧ d = .real) ∨ FP.Software.fp32Context p d) ∧ x = (1e-05 : ℝ) then epsilon32
      else (FP.Software.scalarFP32 rest cfg).literal p d x
    unary := fun p op x => match p, op with
      | some .fp32, .rsqrt => rsqrt32 x
      | _, _ => rest.unary p op x }

#axiomsClean epsilon32_encoding
end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
