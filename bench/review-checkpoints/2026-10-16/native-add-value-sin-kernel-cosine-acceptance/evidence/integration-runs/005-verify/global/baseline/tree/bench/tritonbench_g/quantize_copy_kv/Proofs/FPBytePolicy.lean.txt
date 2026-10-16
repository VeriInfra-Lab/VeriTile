import bench.tritonbench_g.quantize_copy_kv.Proofs.FPContract

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Undefined conversion selects any byte. Its signed interpretation is used
by the int memory channel; this does not select a native overflow policy. -/
def bytePolicy (u : Word → BitVec 8) : Nat → Word → Int := fun _ x => (u x).toInt

theorem quantized_signed_range (cfg : FP.Config) (u : Word → BitVec 8)
    (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) :
    SignedRange 8 (quantized cfg (bytePolicy u) p s i) := by
  cases hx : toSigned 8 (quotient cfg p s i) with
  | none =>
    simp only [quantized, totalSigned, hx, Option.getD_none, bytePolicy]
    exact ⟨by decide, BitVec.le_toInt _, BitVec.toInt_lt⟩
  | some z =>
    obtain ⟨q, hq, ht, hz⟩ := (toSigned_spec 8 (quotient cfg p s i) z).mp hx
    simpa only [quantized, totalSigned, hx, Option.getD_some] using hz

#axiomsClean quantized_signed_range
end
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
