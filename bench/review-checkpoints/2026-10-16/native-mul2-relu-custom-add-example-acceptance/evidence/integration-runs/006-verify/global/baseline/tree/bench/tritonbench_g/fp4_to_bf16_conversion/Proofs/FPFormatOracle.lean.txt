import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPSoftware

/-! Integer-only result oracle for all physical e2m1 nibbles and e8m0 bytes.
The oracle does not call the source, interpreter, pow or floating multiply. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FormatOracle
open VeriTile.Triton
open FPValue
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
set_option maxRecDepth 32000
set_option maxHeartbeats 40000000

/-- Normalized magnitude exponent of {0.5,1,1.5,2,3,4,6}. Zero is handled
separately. These are format values, not source bit-manipulation steps. -/
def magnitudeExponent (m : Nat) : Int :=
  if m = 1 then -1 else if m < 4 then 0 else if m < 6 then 1 else 2
def magnitudeFraction (m : Nat) : Nat := if m = 3 ∨ m = 5 ∨ m = 7 then 64 else 0

/-- e2m1 × e8m0 directly encoded as bf16. Every finite nonzero product has at
most two significand bits and lies at least at 2^-128, so subnormal encodings
need no tie-breaking approximation. Large exponents overflow to infinity.
The scale255 rule canonicalizes NaN irrespective of the nibble's sign. -/
def expectedBits (n : BitVec 4) (b : BitVec 8) : BitVec 16 :=
  let m := n.toNat % 8
  let sign := if n.toNat < 8 then 0 else 0x8000
  let e := (b.toNat : Int) + magnitudeExponent m
  let frac := magnitudeFraction m
  if b.toNat = 255 then 0x7fc0
  else BitVec.ofNat 16 (sign +
    if m = 0 then 0
    else if e ≤ 0 then (128 + frac) / 2 ^ (1-e).toNat
    else if 255 ≤ e then 0x7f80
    else e.toNat * 128 + frac)

/-- Exhaustive kernel check of all 16×256 physical value pairs. The left side
is the existing complete kernel contract's scalar profile; the right side is
the independent integer format oracle above. This is not sampling. -/
theorem all_nibble_scale_pairs : ∀ n : BitVec 4, ∀ b : BitVec 8,
    (bf16Word (software.mulBF16 (embed ⟨Bits.expectedBits n⟩)
      (scaleValue software b.toNat))).bits = expectedBits n b := by
  decide +kernel

theorem boundary_vectors :
    expectedBits 1 0 = 0x0020 ∧ expectedBits 9 0 = 0x8020 ∧
    expectedBits 0 254 = 0 ∧ expectedBits 8 254 = 0x8000 ∧
    expectedBits 7 254 = 0x7f80 ∧ expectedBits 15 254 = 0xff80 ∧
    expectedBits 0 255 = 0x7fc0 ∧ expectedBits 15 255 = 0x7fc0 := by decide +kernel

#specNonCircular expectedBits avoiding [FPValue.source, FP.Structural.exec,
  FPValue.softwarePower, FPValue.scaleValue, FP.mul, FP.round]
#axiomsClean all_nibble_scale_pairs
#axiomsClean boundary_vectors
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FormatOracle
