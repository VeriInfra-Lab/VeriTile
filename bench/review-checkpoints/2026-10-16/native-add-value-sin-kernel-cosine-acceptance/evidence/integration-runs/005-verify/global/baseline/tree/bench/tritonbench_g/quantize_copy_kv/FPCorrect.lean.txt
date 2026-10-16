import bench.tritonbench_g.quantize_copy_kv.Proofs.FPShape
import bench.tritonbench_g.quantize_copy_kv.Proofs.FPBytePolicy

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv
open VeriTile.Triton FP.Structural FP.IntegerConversion
open QuantizeKvTransform TypedQuantizer
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete original per-destination stage: all columns, true half input
and scale payloads, fp32 division, checked signed-byte conversion, both
readbacks and the entire memory frame. Every input bit pattern is covered.
Undefined conversion may choose any byte; no native policy is presumed.
The maxnum reduction schedule/NaN payload and arithmetic promotion are the explicit
software profile, whose GPU lowering is a separate binding obligation. -/
specification quantize_copy_kv_machine_correctness (rest : Algebra Word) (cfg : FP.Config)
    (u : Word → BitVec 8) (p : Parameters) (s : State Word) (hw : 0 < p.width)
    (hregions : p.output ≠ p.scales)
    (hinj : Function.Injective (outputOffset p s))
    (hscale : Function.Injective (scaleOffset p s)) :
    ∃ t, FP.Structural.exec (FPProfile.backend rest cfg (bytePolicy u)) (source p) s = some t ∧
      t.mem = (written cfg (bytePolicy u) p s).mem ∧
      (∀ i : TileIndex [p.height, p.width], TypedQuantizer.active p i →
        t.mem p.output (outputOffset p s i) = .mk .int (quantized cfg (bytePolicy u) p s i) ∧
        SignedRange 8 (quantized cfg (bytePolicy u) p s i)) ∧
      (∀ i : TileIndex [p.height, 1], i.1.val < p.heads →
        t.mem p.scales (scaleOffset p s i) = .mk .fp16 (scale cfg p s i.1)) ∧
      (∀ r o,
        (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], TypedQuantizer.active p i → o ≠ outputOffset p s i) →
        (r ≠ p.scales ∨ ∀ i : TileIndex [p.height, 1], i.1.val < p.heads → o ≠ scaleOffset p s i) →
        t.mem r o = s.mem r o) := by
  refine ⟨(execution rest cfg (bytePolicy u) p s hw).val,
    (execution rest cfg (bytePolicy u) p s hw).property,
    execution_memory rest cfg (bytePolicy u) p s hw, ?_, ?_, ?_⟩
  · intro i hi
    rw [execution_memory]
    exact ⟨value_readback cfg (bytePolicy u) p s i hi hregions hinj,
      quantized_signed_range cfg u p s i⟩
  · intro i hi
    rw [execution_memory]
    exact scale_readback cfg (bytePolicy u) p s i hi hscale
  · intro r o hv hs
    rw [execution_memory]
    exact frame cfg (bytePolicy u) p s r o hv hs

/-- Strengthening on precisely the active conversion domain: every stored
value is toward-zero truncation in signed-int8 range, and changing arbitrary
undefined bytes (including all padded-head conversions) cannot change any
memory cell. There is no condition on inactive heads or on padding count. -/
specification quantize_copy_kv_defined_conversion (rest : Algebra Word) (cfg : FP.Config)
    (u : Word → BitVec 8) (p : Parameters) (s : State Word) (hw : 0 < p.width)
    (hdefined : InputDefined cfg p s) (hregions : p.output ≠ p.scales)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ t, FP.Structural.exec (FPProfile.backend rest cfg (bytePolicy u)) (source p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], TypedQuantizer.active p i →
        ∃ q z, (quotient cfg p s i).toRat? = some q ∧ TowardZero q z ∧ SignedRange 8 z ∧
          t.mem p.output (outputOffset p s i) = .mk .int z) ∧
      (∀ v : Word → BitVec 8, ∃ other,
        FP.Structural.exec (FPProfile.backend rest cfg (bytePolicy v)) (source p) s = some other ∧
        other.mem = t.mem) := by
  refine ⟨(execution rest cfg (bytePolicy u) p s hw).val,
    (execution rest cfg (bytePolicy u) p s hw).property, ?_, ?_⟩
  · intro i hi
    obtain ⟨q, hq, ht, hr⟩ := quantized_defined cfg (bytePolicy u) p s i (hdefined i hi)
    refine ⟨q, quantized cfg (bytePolicy u) p s i, hq, ht, hr, ?_⟩
    rw [execution_memory]
    exact value_readback cfg (bytePolicy u) p s i hi hregions hinj
  · intro v
    exact ⟨(execution rest cfg (bytePolicy v) p s hw).val,
      (execution rest cfg (bytePolicy v) p s hw).property,
      policy_independent rest cfg (bytePolicy v) (bytePolicy u) p s hw hdefined⟩

#axiomsClean quantize_copy_kv_machine_correctness
#axiomsClean quantize_copy_kv_defined_conversion
end
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv
