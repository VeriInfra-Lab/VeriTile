import bench.tritonbench_g.int8_quantization.Proofs.TypedShape
import bench.tritonbench_g.int8_quantization.Proofs.TypedSoftware
namespace VeriTile.Bench.TritonBenchG.Int8Quantization.FPCorrect
open VeriTile.Triton FP.Structural TypedScaled
noncomputable section
/-- Both outputs of the accepted supplied-scale, pre-rounding stage. -/
specification per_block_scaled_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hregions : p.out ≠ p.scaleOut) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ i, row p s i < p.rows → t.mem p.out (offset p s i) = .mk .real (value M p s i)) ∧
      t.mem p.scaleOut (scaleOffset p s) = .mk .real (scale M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) :=
  source_correct M p s hregions

/-- Concrete per-operation fp32 arithmetic for the K preScale=1 profile. -/
specification per_block_scaled_k_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hregions : p.out ≠ p.scaleOut) (hscale : p.preScale = 1) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (kernel p) s = some t ∧
      (∀ i, row p s i < p.rows → t.mem p.out (offset p s i) = .mk .real (roundedValue cfg p s i)) ∧
      t.mem p.scaleOut (scaleOffset p s) = .mk .real ((s.mem p.suppliedScale (scaleOffset p s)).read .real) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hs, hf⟩ := source_correct (FP.Software.scalarFP32 rest cfg) p s hregions
  refine ⟨t, he, ?_, ?_, hf⟩
  · intro i hi
    simpa [value_software rest cfg p s hscale] using hv i hi
  · simpa [scale, FP.Software.scalarFP32] using hs
#axiomsClean per_block_scaled_fp32
#axiomsClean per_block_scaled_k_software
end
end VeriTile.Bench.TritonBenchG.Int8Quantization.FPCorrect
