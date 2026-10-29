import VeriTile.Triton.Random.Proofs.PackRoundedProof
import VeriTile.Triton.Random.Proofs.SignificandBounds

namespace VeriTile.Triton.Random.RoundBridge
open Dyadic

set_option maxRecDepth 3000
set_option maxHeartbeats 2000000

theorem round_zero_decode :
    (FP.round .fp32 FP.Config.ieeeValue 0).decode = .finite false 0 := by
  decide +kernel

/-- Canonical fp32 nearest-even rounding agrees with 24-bit integer rounding
for every nonnegative dyadic in the range needed by both random-conversion
stages, including zero and the carry boundary. -/
theorem round_dyadic_decode (n k : Nat) (hnmax : n < 2^55) (hk : k ≤ 55) :
    (FP.round .fp32 FP.Config.ieeeValue ((n:Rat)/2^k)).decode =
      .finite false ((round24 n:Rat)/2^k) := by
  by_cases hn0 : n = 0
  · subst n
    have h0 : round24 0 = 0 := by decide +kernel
    simp only [h0,Nat.cast_zero,zero_div]
    exact round_zero_decode
  · have hn : 0 < n := by omega
    have hqpos : (0:Rat) < (n:Rat)/2^k := by positivity
    have hlog : n.log2 < 55 := (Nat.log2_lt hn0).mpr hnmax
    rw [round_nonnegative _ hqpos.le, if_neg (ne_of_gt hqpos),
      dyadic_roundStep n k hn (by omega), dyadic_significand_argument n k]
    have hs := normalized_significand n hn
    dsimp only at hs
    rw [decode_packRounded ((n.log2:Int)-k-23) _ (by omega) (by omega) hs.1 hs.2.1]
    apply congrArg (FP.Decoded.finite false)
    have he : (n.log2:Int)-k-23 = ((n.log2:Int)-23)-k := by omega
    have hscale : FP.pow2 ((n.log2:Int)-k-23) =
        FP.pow2 ((n.log2:Int)-23) / (2:Rat)^k := by
      rw [he,pow2_eq_zpow,pow2_eq_zpow,
        zpow_sub₀ (by norm_num : (2:Rat) ≠ 0),zpow_natCast]
    rw [hscale, ←mul_div_assoc, hs.2.2]

end VeriTile.Triton.Random.RoundBridge
