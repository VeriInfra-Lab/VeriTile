import bench.tritonbench_g.fp4_to_bf16.Proofs.HostCorrect
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.RealSpec
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- Independent e2m1 definition: sign, two exponent bits (bias 1), one fraction
bit, including exponent-zero subnormals. It does not inspect the implementation. -/
def e2m1 (q : Bits.Nibble) : ℝ :=
  let u := q.toNat % 8
  let e := u / 2
  let m : ℝ := (u % 2 : Nat)
  let magnitude := if e = 0 then m / 2 else (1 + m / 2) * 2 ^ (e - 1)
  if q.toNat < 8 then magnitude else -magnitude

/-- Finite observation; the separate exact-bit theorem retains zero's sign. -/
def finiteReal (x : FP.Value .bf16) : Option ℝ :=
  match x.decode with
  | .finite sign q => some (if sign then -(q : ℝ) else (q : ℝ))
  | _ => none

theorem magnitude_formula (q : Bits.Nibble) :
    (if decide (8 ≤ q.toNat) then -(Bits.magnitude q : ℝ) else (Bits.magnitude q : ℝ)) =
      e2m1 q := by
  have h := q.isLt
  interval_cases hq : q.toNat <;>
    norm_num [e2m1, Bits.magnitude, hq]

theorem expected_real (q : Bits.Nibble) :
    finiteReal (⟨Bits.expectedBits q⟩ : FP.Value .bf16) = some (e2m1 q) := by
  have hb := (Bits.all_nibbles q).2.1
  have hd := (Bits.all_nibbles q).2.2
  have he : Bits.decoded q = (⟨Bits.expectedBits q⟩ : FP.Value .bf16) := by
    cases h : Bits.decoded q
    simp_all
  rw [he] at hd
  simp only [finiteReal, hd, magnitude_formula]

theorem zeros : e2m1 0 = 0 ∧ e2m1 8 = 0 ∧
    Bits.expectedBits 0 = 0x0000 ∧ Bits.expectedBits 8 = 0x8000 := by
  have h0 : (0 : Bits.Nibble).toNat = 0 := rfl
  have h8 : (8 : Bits.Nibble).toNat = 8 := rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [e2m1, h0]
  · norm_num [e2m1, h8]
  · decide +kernel
  · decide +kernel

#specNonCircular e2m1 avoiding [FPValue.source, FP.Structural.exec, Host.run]
#axiomsClean magnitude_formula
#axiomsClean expected_real
#axiomsClean zeros
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.RealSpec
