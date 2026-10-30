import bench.tritonbench_g.apply_penalty.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum

namespace VeriTile.Bench.Repairs.ApplyPenalty
open VeriTile.Torch

theorem checkInputs_iff (a : Arguments) : checkInputs a = .ok () ↔ InputContract a := by
  classical
  have fields : InputContract a ↔
      (∀ r, TensorValid a r) ∧ (a.tensor 0).shape = [a.batches,a.vocab] ∧
      (0 ≤ a.hint ∧ a.hint ≤ 65536) ∧
      (∀ r : Fin 7, 1 ≤ r.val → r.val < 4 → a.batches ≤ a.capacity r) ∧
      (0 < a.batches → a.batches+1 ≤ a.capacity 6) ∧
      (∀ r : Fin 7, 0 < r.val → DisjointLogits a r) ∧
      (∀ b, b < a.batches → WindowValid a b) := by
    constructor
    · intro h; exact ⟨h.tensors,h.shape,h.hint,h.penalties,h.cumsum,h.disjoint,h.windows⟩
    · rintro ⟨ht,hs,hh,hp,hc,hd,hw⟩; exact ⟨ht,hs,hh,hp,hc,hd,hw⟩
  rw [fields]
  simp [checkInputs,ite_eq_iff]

theorem block_bounds (a : Arguments) (h : InputContract a) :
    512 ≤ a.block ∧ a.block ≤ 65536 ∧ a.hint.toNat ≤ a.block := by
  have hint := h.hint
  have hn : a.hint.toNat ≤ 65536 := by omega
  have hl : (a.hint.toNat-1).log2 < 16 := by
    by_cases hz : a.hint.toNat-1=0
    · simp [hz]
    · exact (Nat.log2_lt hz).mpr (by norm_num; omega)
  have low : 9 ≤ max 9 ((a.hint.toNat-1).log2+1) := le_max_left _ _
  have high : max 9 ((a.hint.toNat-1).log2+1) ≤ 16 := by omega
  have cover := Nat.lt_log2_self (n:=a.hint.toNat-1)
  have power : 2 ^ ((a.hint.toNat-1).log2+1) ≤
      2 ^ max 9 ((a.hint.toNat-1).log2+1) :=
    Nat.pow_le_pow_right (by decide) (le_max_right _ _)
  change 2^9 ≤ 2 ^ max 9 ((a.hint.toNat-1).log2+1) ∧
    2 ^ max 9 ((a.hint.toNat-1).log2+1) ≤ 2^16 ∧
    a.hint.toNat ≤ 2 ^ max 9 ((a.hint.toNat-1).log2+1)
  exact ⟨Nat.pow_le_pow_right (by decide) low,
    Nat.pow_le_pow_right (by decide) high, by omega⟩

theorem window_order (a : Arguments) (h : InputContract a) (b : Nat) (hb : b < a.batches) :
    a.start b ≤ a.stop b := by
  have hw := h.windows b hb
  exact Int.toNat_le_toNat hw.2.1

theorem capacity_logits (a : Arguments) (h : InputContract a) :
    a.capacity 0 = a.batches*a.vocab := by
  simp [Arguments.capacity,TensorMetadata.numel,h.shape]

theorem coordinate_index (v b c t u : Nat) (ht : t < v) (hu : u < v)
    (same : b*v+t=c*v+u) : b=c ∧ t=u := by
  have tok := congrArg (fun n => n % v) same
  simp only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt ht,Nat.mod_eq_of_lt hu] at tok
  have products : b*v=c*v := by omega
  exact ⟨Nat.mul_right_cancel (by omega : 0 < v) products,tok⟩

theorem offset_bound (a : Arguments) (h : InputContract a) (b k : Nat)
    (hb : b < a.batches) (hk : a.start b ≤ k) (he : k < a.stop b) :
    b*a.vocab+a.token k < a.capacity 0 := by
  rw [capacity_logits a h]
  have ht := ((h.windows b hb).2.2.2.2.1 k hk he).2.1
  calc b*a.vocab+a.token k < b*a.vocab+a.vocab := Nat.add_lt_add_left ht _
    _ = (b+1)*a.vocab := by simp [Nat.add_mul]
    _ ≤ a.batches*a.vocab := Nat.mul_le_mul_right _ hb

theorem index_bounds (a : Arguments) (h : InputContract a) (b lane : Nat)
    (hb : b < a.batches) (hl : lane < a.block) :
    b < 2^31 ∧ a.start b+lane < 2^63 ∧
      (a.start b+lane < a.stop b →
        a.start b+lane < a.capacity 4 ∧ a.start b+lane < a.capacity 5 ∧
        b*a.vocab+a.token (a.start b+lane) < 2^31) := by
  have hw := h.windows b hb
  have hp := h.penalties 1 (by decide) (by decide)
  have hpbound : a.capacity 1 < 2^31 := (h.tensors 1).2.2.2.2.2.2.2.1
  have hibound : a.capacity 4 < 2^31 := (h.tensors 4).2.2.2.2.2.2.2.1
  have hobound : a.capacity 0 < 2^31 := (h.tensors 0).2.2.2.2.2.2.2.1
  have hs := window_order a h b hb
  have stopbound := hw.2.2.1
  have block := (block_bounds a h).2.1
  refine ⟨by omega,by omega,?_⟩
  intro active
  have ho := offset_bound a h b (a.start b+lane) hb (by omega) active
  exact ⟨by omega,by omega,by omega⟩

theorem signed64 (n : Nat) (h : n < 2^63) :
    (BitVec.ofNat 64 n).toInt = (n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

/-- Widen before adding the lane. Even a masked-off lane has no signed wrap. -/
theorem native_mask (a : Arguments) (h : InputContract a) (b lane : Nat)
    (hb : b < a.batches) (hl : lane < a.block) :
    (BitVec.ofNat 64 (a.start b) + BitVec.ofNat 64 lane).slt
      (BitVec.ofNat 64 (a.stop b)) = decide (a.start b+lane < a.stop b) := by
  have bounds := index_bounds a h b lane hb hl
  have hs : a.stop b < 2^63 := by
    have hw := (h.windows b hb).2.2.1
    have hc : a.capacity 4 < 2^31 := (h.tensors 4).2.2.2.2.2.2.2.1
    omega
  rw [BitVec.ofNat_add_ofNat,BitVec.slt_eq_decide,signed64 _ bounds.2.1,signed64 _ hs]
  simp only [Int.ofNat_lt]

theorem extend_nonnegative (word : BitVec 32) (nonnegative : 0≤word.toInt) :
    word.signExtend 64=BitVec.ofNat 64 word.toInt.toNat := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have bound := BitVec.toInt_lt (x:=word)
  rw [signed64 _ (by norm_num at bound ⊢; omega),Int.toNat_of_nonneg nonnegative]

/-- The actual signed int32 cumsum loads, sign extension, int64 lane addition
and comparison agree with the natural mask used by both complete proofs. -/
theorem loaded_native_mask (a : Arguments) (h : InputContract a) (b lane : Nat)
    (hb : b<a.batches) (hl : lane<a.block) :
    ((a.cumsum b).signExtend 64+BitVec.ofNat 64 lane).slt
      ((a.cumsum (b+1)).signExtend 64)=decide (a.start b+lane<a.stop b) := by
  have hw := h.windows b hb
  rw [extend_nonnegative _ hw.1,extend_nonnegative _ (le_trans hw.1 hw.2.1)]
  exact native_mask a h b lane hb hl

/-- The natural input channel retains the exact admitted signed-int32 word.
This is an integer theorem, not a floating experiment or an exact FP32 cast. -/
theorem nonnegative_word_roundtrip (word : BitVec 32) (h : 0 ≤ word.toInt) :
    BitVec.ofNat 32 word.toInt.toNat = word := by
  rw [BitVec.toInt_eq_toNat_of_lt (BitVec.toInt_pos_iff.mp h)]
  simp

theorem count_word_roundtrip (a : Arguments) (h : InputContract a) (b k : Nat)
    (hb : b < a.batches) (hs : a.start b ≤ k) (he : k < a.stop b) :
    BitVec.ofNat 32 (a.count k) = a.counts k :=
  nonnegative_word_roundtrip _ ((h.windows b hb).2.2.2.2.1 k hs he).2.2

#axiomsClean checkInputs_iff
#axiomsClean block_bounds
#axiomsClean offset_bound
#axiomsClean index_bounds
#axiomsClean native_mask
#axiomsClean loaded_native_mask
#axiomsClean nonnegative_word_roundtrip
#axiomsClean count_word_roundtrip
end VeriTile.Bench.Repairs.ApplyPenalty
