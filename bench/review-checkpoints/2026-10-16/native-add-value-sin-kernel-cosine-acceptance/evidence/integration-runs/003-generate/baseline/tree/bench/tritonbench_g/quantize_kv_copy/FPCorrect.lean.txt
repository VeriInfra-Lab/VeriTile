import bench.tritonbench_g.quantize_kv_copy.Proofs.TypedShape
import bench.tritonbench_g.quantize_kv_copy.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy
open VeriTile.Triton FP.Structural FP.IntegerConversion TypedQuantize
noncomputable section
local instance : Inhabited TypedQuantize.Word := ⟨FP.Value.zero .fp32⟩

/-- Original grouped quantizer in the supplied fp32-input/fp32-scale/signed-int8
profile: exact encoded maximum, individually rounded scale and quotient,
finite toward-zero integer conversion, both readbacks and memory frame.
The reduction/software configuration is explicit; native GPU lowering is a
separate binding obligation. No condition is imposed on padded groups. -/
specification quantize_kv_group_fp32_int8_correctness (rest : Algebra TypedQuantize.Word) (cfg : FP.Config) (u : Nat → TypedQuantize.Word → Int)
    (p : TypedQuantize.Parameters) (s : State TypedQuantize.Word) (hw : 0 < p.width) (h : TypedQuantize.InputDefined cfg p s)
    (hregions : p.output ≠ p.scales)
    (hinj : ∀ a b, TypedQuantize.active p a → TypedQuantize.active p b → TypedQuantize.outputOffset p s a = TypedQuantize.outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (TypedQuantize.backend rest cfg u) (TypedQuantize.source p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], TypedQuantize.active p i →
        t.mem p.output (TypedQuantize.outputOffset p s i) = .mk .int (TypedQuantize.expected cfg p s i) ∧
        ∃ q, (TypedQuantize.quotient cfg p s i).toRat? = some q ∧
          TowardZero q (TypedQuantize.expected cfg p s i) ∧ SignedRange 8 (TypedQuantize.expected cfg p s i)) ∧
      (∀ i : TileIndex [p.height], i.1.val < p.groups →
        t.mem p.scales (TypedQuantize.scaleOffset p s i) = .mk .real (TypedQuantize.scale cfg p s i.1) ∧
        MatrixReduction.MachineMax.IsMaximum .fp32
          (fun d : Fin p.width => (TypedQuantize.loaded p s (i.1, d, PUnit.unit)).abs.bits)
          (TypedQuantize.rowMaximum p s i.1).bits) ∧
      (∀ r o,
        (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], TypedQuantize.active p i → o ≠ TypedQuantize.outputOffset p s i) →
        (r ≠ p.scales ∨ ∀ i : TileIndex [p.height], i.1.val < p.groups → o ≠ TypedQuantize.scaleOffset p s i) →
        t.mem r o = s.mem r o) :=
  TypedQuantize.correctness rest cfg u p s hw h hregions hinj

#axiomsClean quantize_kv_group_fp32_int8_correctness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvCopy
