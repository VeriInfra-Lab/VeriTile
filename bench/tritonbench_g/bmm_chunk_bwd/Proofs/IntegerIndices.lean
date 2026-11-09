import VeriTile.Triton.Semantics.Integer
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.Sources.Mathematical_RealMath

/-! Pure signed index identities for the original casts and cdiv loops.
Negative chunk limits stay signed; converting the loop count or an active-lane
predicate to natural coordinates does not change any reached memory access. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.IntegerIndices

theorem nat_lt_signed_sub (x c b : Nat) (length : Int) :
    (x : Int) < length - (c : Int) * b ↔ x < length.toNat - c*b := by
  rw [← Int.natCast_mul]
  omega

theorem signed_blocks (length : Int) (width : Nat) :
    ((length + ((width : Int) - 1)).tdiv width).toNat =
      RealMath.blocks length.toNat width := by
  by_cases hw : width = 0
  · subst width
    simp [RealMath.blocks]
  have hwpos : 0 < (width : Int) := by omega
  by_cases hl : 0 ≤ length
  · have hc : ((length.toNat + width - 1 : Nat) : Int) = length + ((width : Int) - 1) := by
      rw [Nat.cast_sub (by omega : 1 ≤ length.toNat + width)]
      simp [Int.toNat_of_nonneg hl]
      omega
    rw [← hc, ← Int.ofNat_tdiv]
    rfl
  · have hn : (length + ((width : Int) - 1)).tdiv width ≤ 0 := by
      calc
        _ ≤ ((width : Int) - 1).tdiv width := Int.tdiv_le_tdiv hwpos (by omega)
        _ = 0 := Int.tdiv_eq_zero_of_lt (by omega) (by omega)
    rw [Int.toNat_of_nonpos hn]
    simp [RealMath.blocks, Int.toNat_of_nonpos (by omega : length ≤ 0), Nat.div_eq_of_lt (by omega : width - 1 < width)]

theorem natural_blocks (length width : Nat) :
    ((length : Int) + (width - 1 : Nat)).tdiv width = (RealMath.blocks length width : Int) := by
  by_cases hw : width = 0
  · subst width
    simp [RealMath.blocks]
  have h : length + (width - 1) = length + width - 1 := by omega
  rw [← Int.natCast_add, ← Int.ofNat_tdiv, h]
  rfl

theorem head_cast (a b : Nat) :
    (a : Int) - (a / b : Nat) * (b : Int) = (a - a / b * b : Nat) := by
  rw [← Int.natCast_mul, Nat.cast_sub (Nat.div_mul_le_self a b)]

theorem remainder_cast (a b : Nat) :
    (a : Int) - (b : Int) * ((a : Int) / b) = (a % b : Nat) := by
  rw [← Int.emod_def, Int.natCast_emod]

theorem head_sub_cast (a b : Nat) :
    (a : Int) - (a / b * b : Nat) = (a - a / b * b : Nat) := by
  rw [Int.natCast_mul]
  exact head_cast a b

theorem remainder_sub_cast (a b : Nat) :
    (a : Int) - (b * (a / b) : Nat) = (a % b : Nat) := by
  rw [Int.natCast_mul, Int.natCast_ediv]
  exact remainder_cast a b

theorem source_min (x y : Int) : (if x < y then x else y) = min x y := by
  split_ifs with h
  · exact (min_eq_left (le_of_lt h)).symm
  · exact (min_eq_right (le_of_not_gt h)).symm

#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.IntegerIndices
