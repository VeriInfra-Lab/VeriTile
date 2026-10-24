import bench.tritonbench_g.rowwise_quantization_triton.Proofs.TypedShape
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.TypedSoftware
namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.FPCorrect
open VeriTile.Triton FP.Structural TypedScaled
noncomputable section
/-- The accepted pre-llrint stage: typed floating operations, independent
initial-memory values, both stores, and a complete memory frame. -/
specification rowwise_scaled_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hregions : p.out ≠ p.maxOut) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], i.1.val < p.columns →
        t.mem p.out (TypedScaled.offset p s i) = .mk .real (scaledValue M p s i)) ∧
      t.mem p.maxOut (s.pids 0) = .mk .fp16 (maximumStored M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) :=
  source_correct M p s hwidth hregions
/-- Explicit binary32 values for the row and direct binary16 rounding for
its maximum. The explicit maxnum reduction schedule and NaN/zero profile
are part of this software model; native lowering remains a separate binding. -/
specification rowwise_scaled_software_correctness [Inhabited Word]
    (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hwidth : 0 < p.width) (hregions : p.out ≠ p.maxOut) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg) (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], i.1.val < p.columns →
        t.mem p.out (TypedScaled.offset p s i) = .mk .real (bitValue cfg p s i)) ∧
      t.mem p.maxOut (s.pids 0) = .mk .fp16 (bitMaximumStored cfg p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hm, hf⟩ := source_correct (bitBackend rest cfg) p s hwidth hregions
  refine ⟨t, he, ?_, ?_, hf⟩
  · intro i hi
    simpa only [value_bits] using hv i hi
  · simpa only [maximum_stored_bits] using hm

#axiomsClean rowwise_scaled_typed_correctness
#axiomsClean rowwise_scaled_software_correctness
end
end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.FPCorrect
