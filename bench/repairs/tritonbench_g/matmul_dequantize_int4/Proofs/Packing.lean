import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Packing
abbrev Word := BitVec 32
abbrev Nibble := BitVec 4
/-- Independent packed-word target: little-endian concatenation of eight
unsigned nibbles. Signed int32 storage carries exactly these low 32 bits. -/
def packed (q : Fin 8 → Nibble) : Word :=
  q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0
/-- Actual helper's zero-initialized wrapping int32 shift-and-add recurrence. -/
def accumulate (q : Fin 8 → Nibble) : Nat → Word
  | 0 => 0
  | j+1 => accumulate q j + ((q ⟨j%8,Nat.mod_lt _ (by decide)⟩).zeroExtend 32 <<< (j*4))
def nibble (w : Word) (i : Fin 8) : Nibble := w.extractLsb' (i.val*4) 4

theorem accumulate_packed (q : Fin 8 → Nibble) : accumulate q 8=packed q := by
  change (((((((((0 : Word)+((q 0).zeroExtend 32 <<< 0))+((q 1).zeroExtend 32 <<< 4))+((q 2).zeroExtend 32 <<< 8))+((q 3).zeroExtend 32 <<< 12))+((q 4).zeroExtend 32 <<< 16))+((q 5).zeroExtend 32 <<< 20))+((q 6).zeroExtend 32 <<< 24))+((q 7).zeroExtend 32 <<< 28))=(q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0)
  have appendNat {m n : Nat} (a : BitVec m) (b : BitVec n) :
      (a ++ b).toNat = a.toNat * 2^n + b.toNat := by
    rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt b.isLt, Nat.shiftLeft_eq]
  have h0 := (q 0).isLt
  have h1 := (q 1).isLt
  have h2 := (q 2).isLt
  have h3 := (q 3).isLt
  have h4 := (q 4).isLt
  have h5 := (q 5).isLt
  have h6 := (q 6).isLt
  have h7 := (q 7).isLt
  have hz : ((0 : Word).toNat) = 0 := rfl
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_add, BitVec.toNat_shiftLeft, BitVec.zeroExtend,
    BitVec.toNat_setWidth, hz, appendNat, Nat.shiftLeft_eq]
  norm_num at *
  omega
theorem nibble_packed (q : Fin 8 → Nibble) (i : Fin 8) : nibble (packed q) i=q i := by
  fin_cases i
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 0 4=q 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 4 4=q 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 8 4=q 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 12 4=q 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤12), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 16 4=q 4
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤16), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤12), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 20 4=q 5
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤20), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤16), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤12), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 24 4=q 6
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤24), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤20), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤16), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤12), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0) : Word).extractLsb' 28 4=q 7
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 4≤28), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤24), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤20), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤16), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤12), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤8), BitVec.extractLsb'_append_eq_of_le (by decide : 4≤4)]
    exact BitVec.extractLsb'_eq_self

theorem signed_storage_roundtrip (q : Fin 8 → Nibble) : BitVec.ofInt 32 (packed q).toInt=packed q := by
  simp
/-- Keeping an arbitrary fresh seed is observable even when all eight
quantized values are zero. This is the original source defect. -/
def seeded (seed : Word) (q : Fin 8 → Nibble) : Nat → Word
  | 0 => seed
  | j+1 => seeded seed q j + ((q ⟨j%8,Nat.mod_lt _ (by decide)⟩).zeroExtend 32 <<< (j*4))
theorem seed_survives (seed : Word) : seeded seed (fun _ => 0) 8=seed := by simp [seeded]
theorem zero_seed_correct (q : Fin 8 → Nibble) : seeded 0 q 8=packed q := accumulate_packed q
#specNonCircular packed avoiding [accumulate,seeded]
#axiomsClean accumulate_packed
#axiomsClean nibble_packed
#axiomsClean signed_storage_roundtrip
#axiomsClean seed_survives
#axiomsClean zero_seed_correct
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Packing
