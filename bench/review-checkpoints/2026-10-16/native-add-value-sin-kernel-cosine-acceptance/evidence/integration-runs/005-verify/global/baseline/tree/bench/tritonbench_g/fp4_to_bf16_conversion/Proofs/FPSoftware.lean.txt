import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
set_option maxRecDepth 32000
set_option maxHeartbeats 10000000

/-- Explicit software power-of-two profile: exact rational integer power,
then fp32 IEEE rounding. This is not an assertion about native __nv_powif. -/
def softwarePower (n : Int) : Word := FP.round .fp32 .ieeeValue (FP.pow2 n)
def bf16Word (x : Word) : FP.Value .bf16 := ⟨x.bits.truncate 16⟩
def software : Primitives :=
  { pow2 := softwarePower
    mulBF16 := fun x y => embed (FP.mul .ieeeValue (bf16Word x) (bf16Word y)) }

/-- Independent e8m0 interpretation: scale 0 is 2^-127, 1..254 have the
corresponding bf16 exponent field, and 255 is canonical quiet NaN. -/
def e8m0Bits (b : BitVec 8) : BitVec 16 :=
  if b.toNat=255 then 0x7fc0 else if b.toNat=0 then 0x0040
  else BitVec.ofNat 16 (b.toNat*128)

theorem byte_exponent_exact : ∀ b : BitVec 8,
    scaleExponent b.toNat = (b.toNat : Int)-127 := by decide +kernel

theorem all_scale_bytes : ∀ b : BitVec 8,
    scaleValue software b.toNat = embed ⟨e8m0Bits b⟩ := by decide +kernel

/-- Includes subnormal output, signed zero, overflow, scale NaN and the
otherwise hidden power overflow before the NaN override. -/
theorem software_vectors :
    (softwarePower (-127)).bits = 0x00400000 ∧
    (softwarePower 128).bits = 0x7f800000 ∧
    (bf16Word (scaleValue software 255)).bits = 0x7fc0 ∧
    (bf16Word (software.mulBF16 (embed ⟨0x3f00⟩) (scaleValue software 0))).bits = 0x0020 ∧
    (bf16Word (software.mulBF16 (embed ⟨0x8000⟩) (scaleValue software 254))).bits = 0x8000 ∧
    (bf16Word (software.mulBF16 (embed ⟨0x40c0⟩) (scaleValue software 254))).bits = 0x7f80 ∧
    (bf16Word (software.mulBF16 (embed ⟨0⟩) (scaleValue software 255))).bits = 0x7fc0 ∧
    (bf16Word (software.mulBF16 (embed ⟨0⟩) (embed ⟨0x7f80⟩))).bits = 0x7fc0 := by
  decide +kernel

#axiomsClean byte_exponent_exact
#axiomsClean all_scale_bytes
#axiomsClean software_vectors
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
