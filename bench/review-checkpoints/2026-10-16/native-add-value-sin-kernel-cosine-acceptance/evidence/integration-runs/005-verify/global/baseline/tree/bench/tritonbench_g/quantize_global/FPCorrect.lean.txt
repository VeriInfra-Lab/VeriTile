import bench.tritonbench_g.quantize_global.Proofs.TypedShape
namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal.FPCorrect
open VeriTile.Triton FP.Structural TypedScaled
noncomputable section
/-- Actual binary32 arithmetic for the accepted pre-rounding stage. The
unmasked supplied scale load, tail mask, primitive order and frame are kept. -/
specification quantize_global_scaled_fp32 (rest : Algebra Word) (cfg : FP.Config)
    (s : State Word) (X AbsmaxInv Out : RegionName) (N B : Nat) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source X AbsmaxInv Out N B) s = some t ∧
      (∀ i, TypedScaled.offset s B i < N → t.mem Out (TypedScaled.offset s B i) = .mk .real (value cfg s X AbsmaxInv B i)) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, TypedScaled.offset s B i < N → o ≠ TypedScaled.offset s B i) → t.mem r o = s.mem r o) :=
  source_correct rest cfg s X AbsmaxInv Out N B
#axiomsClean quantize_global_scaled_fp32
end
end VeriTile.Bench.TritonBenchG.QuantizeGlobal.FPCorrect
