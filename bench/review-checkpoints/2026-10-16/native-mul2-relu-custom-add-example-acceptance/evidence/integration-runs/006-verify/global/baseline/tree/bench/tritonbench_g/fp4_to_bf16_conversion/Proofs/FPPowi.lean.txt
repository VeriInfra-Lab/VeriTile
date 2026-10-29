import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPSoftware

/-! A software semantics of the actual repaired kernel's lowered integer-power
instruction graph. This file does not assert a PTX/hardware refinement: mul and
reciprocal below use the explicitly named software IEEE profiles. The original
Python pow(fp32,int16) overload is still rejected by Triton 3.8.0. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.Powi
open VeriTile.Triton
open FPValue
set_option maxRecDepth 32000
set_option maxHeartbeats 20000000

def ftz : FP.Config := ⟨.nearestEven, Bool.true, Bool.true⟩

/-- Exponentiation-by-squaring loop seen in the pinned SM90 PTX. The exponent
is shifted after the conditional accumulator multiply; the base is squared only
if another iteration is needed. The optional result exposes fuel exhaustion. -/
def loop : Nat → Nat → Word → Word → Option Word
  | 0, _, _, _ => none
  | fuel+1, e, base, acc =>
      let acc := if e % 2 = 1 then FP.mul ftz acc base else acc
      let next := e / 2
      if next = 0 then some acc
      else loop fuel next (FP.mul ftz base base) acc

/-- The native graph computes a reciprocal with round-to-nearest and no FTZ
modifier when the original exponent is negative. In particular the final
2^-127 result is permitted to be subnormal. -/
def loweredPower (n : Int) : Option Word := do
  let positive ← loop 8 n.natAbs ⟨0x40000000⟩ ⟨0x3f800000⟩
  pure (if n < 0 then FP.div .ieeeValue ⟨0x3f800000⟩ positive else positive)

theorem boundary_values :
    loweredPower (-127) = some ⟨0x00400000⟩ ∧
    loweredPower (-126) = some ⟨0x00800000⟩ ∧
    loweredPower 0 = some ⟨0x3f800000⟩ ∧
    loweredPower 127 = some ⟨0x7f000000⟩ ∧
    loweredPower 128 = some ⟨0x7f800000⟩ := by decide +kernel

/-- Exhaustive, kernel-checked statement for the entire physical scale-byte
domain. It proves termination within eight iterations and exact software bit
equality with direct rational power rounding, including byte255/exponent128.
It is not a native GPU binding and imposes no finite/non-NaN premise. -/
theorem all_scale_exponents : ∀ b : BitVec 8,
    loweredPower ((b.toNat : Int)-127) =
      some (softwarePower ((b.toNat : Int)-127)) := by decide +kernel

theorem exact_source_exponent (b : BitVec 8) :
    loweredPower (scaleExponent b.toNat) = some (softwarePower (scaleExponent b.toNat)) := by
  rw [byte_exponent_exact b]
  exact all_scale_exponents b

/-- Flushing the reciprocal result would change scale byte0. The actual PTX
uses rcp.rn.f32, not rcp.rn.ftz.f32; the distinction is observable. -/
theorem reciprocal_ftz_counterexample :
    FP.div ftz (⟨0x3f800000⟩ : Word) ⟨0x7f000000⟩ = ⟨0⟩ ∧
    FP.div .ieeeValue (⟨0x3f800000⟩ : Word) ⟨0x7f000000⟩ = ⟨0x00400000⟩ := by
  decide +kernel

/-- The byte-domain result must not be generalized to arbitrary signed16
exponents. Computing the positive power first overflows at128; taking its
reciprocal then returns zero, although direct rounding of 2^-128 is nonzero.
The old interpreter permits the non-byte Nat65535 as an arbitrary masked
undef, whose signed16 exponent is -128. We retain that wider old contract and
do not claim this lowered graph refines it on such nonphysical values. -/
theorem outside_byte_domain_counterexample :
    scaleExponent 65535 = -128 ∧
    loweredPower (-128) = some ⟨0⟩ ∧
    (softwarePower (-128)).bits = 0x00200000 := by decide +kernel

#specNonCircular loweredPower avoiding [FPValue.source, FP.Structural.exec, FPValue.softwarePower]
#axiomsClean boundary_values
#axiomsClean all_scale_exponents
#axiomsClean exact_source_exponent
#axiomsClean reciprocal_ftz_counterexample
#axiomsClean outside_byte_domain_counterexample
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.Powi
