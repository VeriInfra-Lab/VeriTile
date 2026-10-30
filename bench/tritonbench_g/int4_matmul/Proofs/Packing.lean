import bench.tritonbench_g.int4_matmul.Kernels
namespace VeriTile.Bench.Repairs.Int4Matmul
set_option maxHeartbeats 4000000
/-- The true int32 arithmetic shift and mask agree with unsigned packed-nibble
extraction for all eight positions, including negative packed words. -/
theorem signed_logical_mask (word : BitVec 32) (slot : Fin 8) :
    (word.sshiftRight (slot.val * 4) &&& (15 : BitVec 32)) =
      ((word >>> (slot.val * 4)) &&& (15 : BitVec 32)) := by
  apply BitVec.eq_of_getElem_eq
  intro i hi
  by_cases hlow : i < 4
  · have hsum : slot.val * 4 + i < 32 := by omega
    simp [BitVec.getElem_sshiftRight, BitVec.getElem_ushiftRight, BitVec.getLsbD_eq_getElem, hsum]
  · have hmask : (15 : Nat).testBit i = Bool.false := by
      interval_cases i <;> decide +kernel
    simp [BitVec.getElem_and, BitVec.getElem_eq_testBit_toNat, BitVec.toNat_ofNat, hmask]

theorem mask_bound (word : BitVec 32) :
    0 ≤ (word &&& (15 : BitVec 32)).toInt ∧ (word &&& (15 : BitVec 32)).toInt ≤ 15 := by
  have hn : (word &&& (15 : BitVec 32)).toNat ≤ 15 := by
    simpa only [BitVec.toNat_and, BitVec.toNat_ofNat, Nat.reduceMod] using
      (Nat.and_le_right (n := word.toNat) (m := 15))
  rw [BitVec.toInt_eq_toNat_of_lt (by omega)]
  omega

theorem difference_no_overflow (a b : BitVec 32) :
    (BitVec.ofInt 32 ((a &&& (15 : BitVec 32)).toInt - (b &&& (15 : BitVec 32)).toInt)).toInt =
      (a &&& (15 : BitVec 32)).toInt - (b &&& (15 : BitVec 32)).toInt := by
  have ha := mask_bound a
  have hb := mask_bound b
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · omega
  · omega


/-- The repaired eight augmenting updates start from specified int32 zero. -/
def packLoop (q : Fin 8 → BitVec 4) : Nat → Word
  | 0 => 0
  | n+1 => packLoop q n + ((q ⟨n%8,Nat.mod_lt _ (by decide)⟩).zeroExtend 32 <<< (4*n))
def packedSpec (q : Fin 8 → BitVec 4) : Word :=
  q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0

theorem append_nat (a : BitVec m) (b : BitVec n) : (a++b).toNat=a.toNat*2^n+b.toNat := by
  rw [BitVec.toNat_append,←Nat.shiftLeft_add_eq_or_of_lt b.isLt,Nat.shiftLeft_eq]

theorem mask15 (n : Nat) : n &&& 15=n%16 := by
  simpa using Nat.and_two_pow_sub_one_eq_mod n 4

theorem packLoop_correct (q : Fin 8 → BitVec 4) : packLoop q 8=packedSpec q := by
  change (((((((((0 : Word) + ((q 0).zeroExtend 32 <<< (0 : Nat))) + ((q 1).zeroExtend 32 <<< (4 : Nat))) + ((q 2).zeroExtend 32 <<< (8 : Nat))) + ((q 3).zeroExtend 32 <<< (12 : Nat))) + ((q 4).zeroExtend 32 <<< (16 : Nat))) + ((q 5).zeroExtend 32 <<< (20 : Nat))) + ((q 6).zeroExtend 32 <<< (24 : Nat))) + ((q 7).zeroExtend 32 <<< (28 : Nat))) = (q 7 ++ q 6 ++ q 5 ++ q 4 ++ q 3 ++ q 2 ++ q 1 ++ q 0)
  have h0 : (q 0).toNat<16 := (q 0).isLt
  have h1 : (q 1).toNat<16 := (q 1).isLt
  have h2 : (q 2).toNat<16 := (q 2).isLt
  have h3 : (q 3).toNat<16 := (q 3).isLt
  have h4 : (q 4).toNat<16 := (q 4).isLt
  have h5 : (q 5).toNat<16 := (q 5).isLt
  have h6 : (q 6).toNat<16 := (q 6).isLt
  have h7 : (q 7).toNat<16 := (q 7).isLt
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_add,BitVec.zeroExtend_eq_setWidth,BitVec.toNat_setWidth,
    BitVec.toNat_shiftLeft,Nat.shiftLeft_eq,append_nat,show (0 : Word).toNat=0 from rfl]
  norm_num
  omega

theorem packed_nat (q : Fin 8 → BitVec 4) : (packedSpec q).toNat=(q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456 := by
  unfold packedSpec
  simp only [append_nat]
  norm_num
  ring

theorem unpack_packed (q : Fin 8 → BitVec 4) (i : Fin 8) :
    (((packedSpec q >>> (4*i.val)) &&& (15 : Word)).toNat)= (q i).toNat := by
  have h0 : (q 0).toNat<16 := (q 0).isLt
  have h1 : (q 1).toNat<16 := (q 1).isLt
  have h2 : (q 2).toNat<16 := (q 2).isLt
  have h3 : (q 3).toNat<16 := (q 3).isLt
  have h4 : (q 4).toNat<16 := (q 4).isLt
  have h5 : (q 5).toNat<16 := (q 5).isLt
  have h6 : (q 6).toNat<16 := (q 6).isLt
  have h7 : (q 7).toNat<16 := (q 7).isLt
  simp only [BitVec.toNat_and,BitVec.toNat_ushiftRight,packed_nat,
    show (15 : Word).toNat=15 from rfl,mask15,Nat.shiftRight_eq_div_pow]
  fin_cases i
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 1) % 16=(q 0).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 16) % 16=(q 1).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 256) % 16=(q 2).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 4096) % 16=(q 3).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 65536) % 16=(q 4).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 1048576) % 16=(q 5).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 16777216) % 16=(q 6).toNat
    omega
  · change (((q 0).toNat * 1 + (q 1).toNat * 16 + (q 2).toNat * 256 + (q 3).toNat * 4096 + (q 4).toNat * 65536 + (q 5).toNat * 1048576 + (q 6).toNat * 16777216 + (q 7).toNat * 268435456) / 268435456) % 16=(q 7).toNat
    omega

theorem signed_nibble (w : Word) (slot : Nat) :
    (w.sshiftRight (slot%8*4) &&& (15 : Word)).toInt=nibble w slot := by
  rw [signed_logical_mask w ⟨slot%8,Nat.mod_lt _ (by decide)⟩]
  rfl

theorem nibble_bounds (w : Word) (slot : Nat) : 0≤nibble w slot ∧ nibble w slot≤15 :=
  mask_bound _
theorem difference_bound (s : Shape) (m : State α) (k c : Nat) :
    -15≤difference s m k c ∧ difference s m k c≤15 := by
  have a := nibble_bounds (m.packed .weight (qAddress s k c)) k
  have b := nibble_bounds (m.packed .zero (zAddress s k c)) c
  unfold difference; omega
#axiomsClean packLoop_correct
#axiomsClean unpack_packed
#axiomsClean signed_nibble
#axiomsClean difference_bound
end VeriTile.Bench.Repairs.Int4Matmul
