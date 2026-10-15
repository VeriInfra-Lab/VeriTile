import bench.tritonbench_g.quantize_kv_transform.Proofs.TypedShape
import bench.tritonbench_g.quantize_kv_transform.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
local instance : Inhabited TypedTransform.Word := ⟨FP.Value.zero .fp32⟩

/-- Original full accepted transform stage for both supplied input formats,
with actual binary16 scale encoding and binary32 division, checked signed-int8
conversion and independent maximum/truncation/readback/frame properties.
Source-format and compute-format operations are distinct; GPU lowering remains
an explicit binding obligation. Padded lanes have no extra numeric restriction. -/
specification quantize_kv_transform_machine_correctness (rest : Algebra TypedTransform.Word) (cfg : FP.Config) (u : Nat → TypedTransform.Word → Int)
    (d : TypedTransform.Storage) (p : TypedTransform.Parameters) (s : State TypedTransform.Word) (hw : 0 < p.width) (h : TypedTransform.InputDefined cfg d p s)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, TypedTransform.active p a → TypedTransform.active p b → TypedTransform.outputOffset p s a = TypedTransform.outputOffset p s b → a = b)
    (hscale : ∀ a b : TileIndex [p.height, 1], a.1.val < p.heads → b.1.val < p.heads →
      TypedTransform.scaleOffset p s a = TypedTransform.scaleOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (MixedBits.backend rest cfg u) (TypedTransform.source d p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], TypedTransform.active p i →
        t.mem p.output (TypedTransform.outputOffset p s i) = .mk .int (TypedTransform.expected cfg d p s i) ∧
        ∃ q, (TypedTransform.quotient cfg d p s i).toRat? = some q ∧
          TowardZero q (TypedTransform.expected cfg d p s i) ∧ SignedRange 8 (TypedTransform.expected cfg d p s i)) ∧
      (∀ i : TileIndex [p.height, 1], i.1.val < p.heads →
        t.mem p.scales (TypedTransform.scaleOffset p s i) = .mk .fp16 (TypedTransform.scale cfg d p s i.1) ∧
        MatrixReduction.MachineMax.IsMaximum .fp32
          (fun j : Fin p.width => (TypedTransform.absPromoted d (TypedTransform.loaded d p s (i.1, j, PUnit.unit))).bits)
          (TypedTransform.maximum d p s i.1).bits) ∧
      (∀ r o,
        (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], TypedTransform.active p i → o ≠ TypedTransform.outputOffset p s i) →
        (r ≠ p.scales ∨ ∀ i : TileIndex [p.height, 1], i.1.val < p.heads → o ≠ TypedTransform.scaleOffset p s i) →
        t.mem r o = s.mem r o) :=
  TypedTransform.correctness rest cfg u d p s hw h hregions hinj hscale

#axiomsClean quantize_kv_transform_machine_correctness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform
