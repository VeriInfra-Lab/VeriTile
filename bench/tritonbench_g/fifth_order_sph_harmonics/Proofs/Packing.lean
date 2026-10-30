import VeriTile.Triton
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Packing
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000

abbrev Word := Fin 4 → BitVec 8

/-- Native little-endian FP32 byte representation, including all NaN payloads
and both signs of zero. This layer performs no numerical rounding. -/
def pack (w : Word) : BitVec 32 := w 3 ++ w 2 ++ w 1 ++ w 0
def unpack (bits : BitVec 32) : Word := fun b=>bits.extractLsb' (8*b.val) 8
def value (w : Word) : FP.Value .fp32 := ⟨pack w⟩
def word (x : FP.Value .fp32) : Word := unpack x.bits

theorem unpack_pack (w : Word) : unpack (pack w)=w := by
  funext b
  fin_cases b
  · change (w 3 ++ w 2 ++ w 1 ++ w 0).extractLsb' 0 8=w 0
    rw [BitVec.extractLsb'_append_eq_right]
  · change (w 3 ++ w 2 ++ w 1 ++ w 0).extractLsb' 8 8=w 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide)]
    exact BitVec.extractLsb'_append_eq_right
  · change (w 3 ++ w 2 ++ w 1 ++ w 0).extractLsb' 16 8=w 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide),BitVec.extractLsb'_append_eq_of_le (by decide)]
    exact BitVec.extractLsb'_append_eq_right
  · change (w 3 ++ w 2 ++ w 1 ++ w 0).extractLsb' 24 8=w 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide),BitVec.extractLsb'_append_eq_of_le (by decide)]
    exact BitVec.extractLsb'_append_eq_left

theorem pack_unpack (bits : BitVec 32) : pack (unpack bits)=bits := by
  norm_num [pack,unpack]
  bv_decide

theorem value_word (x : FP.Value .fp32) : value (word x)=x := by
  cases x
  simp [value,word,pack_unpack]

theorem word_value (w : Word) : word (value w)=w := unpack_pack w

#axiomsClean unpack_pack
#axiomsClean pack_unpack
#axiomsClean value_word
#axiomsClean word_value
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Packing
