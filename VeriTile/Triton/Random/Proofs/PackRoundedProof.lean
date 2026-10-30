import VeriTile.Triton.Random.Proofs.PackDecode

namespace VeriTile.Triton.Random.RoundBridge

set_option maxRecDepth 4000
set_option maxHeartbeats 2000000

/-- The only carry possible after nearest-even rounding of a normalized
significand preserves its exact dyadic value. -/
theorem decode_packRounded (step : Int) (n : Nat)
    (hstep : -149 ≤ step) (hmax : step ≤ 103)
    (hn : 2^23 ≤ n) (hnmax : n ≤ 2^24) :
    (packRounded False step n).decode = .finite false ((n:Rat)*FP.pow2 step) := by
  let carry : Int := if 2^24 ≤ n then 1 else 0
  let m : Nat := if 2^24 ≤ n then n/2 else n
  have hc : 0 ≤ carry ∧ carry ≤ 1 := by dsimp [carry]; split <;> omega
  have hm : 2^23 ≤ m ∧ m < 2^24 := by dsimp [m]; split <;> omega
  have ho : ¬(step+23+carry > 127) := by omega
  have hmlo : ¬m < 2^23 := by omega
  have hsum : step+23+carry+127 = step+carry+150 := by omega
  have hpack : packRounded False step n =
      FP.Value.ofBits .fp32 ((step+carry+150).toNat*2^23+m-2^23) := by
    unfold packRounded
    change (if step+23+carry > 127 then _ else if m < 2^23 then _ else _) = _
    rw [if_neg ho, if_neg hmlo]
    simp only [↓reduceIte, Nat.zero_add]
    rw [hsum]
  rw [hpack, decode_scaled_significand (step+carry) m (by omega) (by omega) hm.1 hm.2]
  apply congrArg (FP.Decoded.finite false)
  by_cases h : 2^24 ≤ n
  · have heq : n = 2^24 := by omega
    subst n
    simp only [carry, m, le_refl, ↓reduceIte]
    rw [pow2_eq_zpow, pow2_eq_zpow,
      zpow_add₀ (by norm_num : (2:Rat) ≠ 0)]
    norm_num
    ring
  · simp only [carry, m, h, ↓reduceIte, Int.add_zero]

end VeriTile.Triton.Random.RoundBridge
