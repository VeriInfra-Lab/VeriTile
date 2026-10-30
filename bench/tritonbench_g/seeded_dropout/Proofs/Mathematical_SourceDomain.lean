import Mathlib.Tactic
import Mathlib.Algebra.Order.Floor.Div

/-!
Source-index domain for the original seeded_dropout kernel. This module uses
only mathematical/BitVec libraries and does not import the VeriTile AST.

The source computes pid, pid * B, arange(0, B), and their lane-wise sum in i32
for the recorded native specialization. The host grid uses Python cdiv, whose
intermediate arithmetic is unbounded; it is not an i32 expression in the kernel.
All lanes, including inactive tail lanes, must have faithful i32 offsets because
the generator is evaluated before the final masked store. Runtime seed and p
are untouched, and this module does not simplify the eager floating division.
-/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.SourceDomain

/-- A sufficient legal one-dimensional arange shape for Triton 3.7.1:
positive power of two, at most its 2^20-element frontend tensor limit. -/
def LegalBlock (B : Nat) : Prop := ∃ k : Nat, k ≤ 20 ∧ B = 2^k

/-- The pinned wrapper's host launch expression. -/
def grid (N B : Nat) : Nat := (N+B-1)/B

/-- Signed i32 representability, independently of any final RNG guard. -/
def I32Rep (value : Int) : Prop := -(2^31 : Int) ≤ value ∧ value < 2^31

theorem legalBlock_fields {B : Nat} (hB : LegalBlock B) :
    0 < B ∧ B ≤ 2^20 ∧ B < 2^31 ∧ B ∣ 2^31 := by
  obtain ⟨k, hk, rfl⟩ := hB
  have hbound : (2:Nat)^k ≤ 2^20 := Nat.pow_le_pow_right (by decide) hk
  exact ⟨by positivity, hbound, lt_of_le_of_lt hbound (by norm_num),
    Nat.pow_dvd_pow 2 (by omega)⟩

/-- Numeric conditions tested by the installed arange/block-shape frontend. -/
theorem legalBlock_arange_checks {B : Nat} (hB : LegalBlock B) :
    0 < B ∧ B ≤ 1048576 ∧ B >>> 32 = 0 ∧ B &&& (B-1) = 0 := by
  have hf := legalBlock_fields hB
  refine ⟨hf.1, by norm_num at hf ⊢; omega, ?_, ?_⟩
  · rw [Nat.shiftRight_eq_div_pow]
    exact Nat.div_eq_of_lt (lt_trans hf.2.2.1 (by norm_num))
  · obtain ⟨k, _, rfl⟩ := hB
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_self]

theorem wrapper_block_legal : LegalBlock 1024 := ⟨10, by decide, by decide⟩

/-- Zero elements launch no programs under the source's host grid expression. -/
theorem grid_zero (B : Nat) : grid 0 B = 0 := by
  change (0 : Nat) ⌈/⌉ B = 0
  simp

theorem grid_le_i32_blocks {N B : Nat} (hB : LegalBlock B) (hN : N < 2^31) :
    grid N B ≤ 2^31/B := by
  have hf := legalBlock_fields hB
  change N ⌈/⌉ B ≤ 2^31/B
  apply (ceilDiv_le_iff_le_mul hf.1).mpr
  rw [Nat.mul_div_cancel' hf.2.2.2]
  exact hN.le

/-- Each launched block starts before N; no block exists when N is zero. -/
theorem launched_block_start_lt {N B pid : Nat} (hB : LegalBlock B)
    (hpid : pid < grid N B) : pid*B < N := by
  have hpos := (legalBlock_fields hB).1
  by_contra h
  have hceil : N ⌈/⌉ B ≤ pid :=
    (ceilDiv_le_iff_le_mul hpos).mpr (by simpa [Nat.mul_comm] using (Nat.le_of_not_gt h))
  change grid N B ≤ pid at hceil
  omega

/-- The entire launched tile fits i32, including lanes masked off by offsets<N.
Divisibility of 2^31 by the legal block size controls the rounded-up tail. -/
theorem launched_lane_lt_i32 {N B pid lane : Nat}
    (hB : LegalBlock B) (hN : N < 2^31)
    (hpid : pid < grid N B) (hlane : lane < B) : pid*B+lane < 2^31 := by
  have hgrid := grid_le_i32_blocks hB hN
  have hblock : (pid+1)*B ≤ (2^31/B)*B :=
    Nat.mul_le_mul_right B (by omega)
  have hd := (legalBlock_fields hB).2.2.2
  rw [Nat.div_mul_cancel hd] at hblock
  calc
    pid*B+lane < pid*B+B := Nat.add_lt_add_left hlane _
    _ = (pid+1)*B := by ring
    _ ≤ 2^31 := hblock

/-- Every integer source operand and each intermediate result is representable.
The final offset bound alone is not used as a substitute for these facts. -/
theorem source_intermediates_i32 {N B pid lane : Nat}
    (hB : LegalBlock B) (hN : N < 2^31)
    (hpid : pid < grid N B) (hlane : lane < B) :
    I32Rep (N:Int) ∧ I32Rep (B:Int) ∧ I32Rep (pid:Int) ∧ I32Rep (lane:Int) ∧
    I32Rep ((pid*B:Nat):Int) ∧ I32Rep ((pid*B+lane:Nat):Int) := by
  have hfields := legalBlock_fields hB
  have hoff := launched_lane_lt_i32 hB hN hpid hlane
  have hstart : pid*B < 2^31 := by omega
  have hpidle : pid ≤ pid*B := by
    simpa using Nat.mul_le_mul_left pid (Nat.succ_le_of_lt hfields.1)
  have hpidbound : pid < 2^31 := lt_of_le_of_lt hpidle hstart
  have hlaneBound : lane < 2^31 := lt_trans hlane hfields.2.2.1
  unfold I32Rep
  norm_num at hN hfields hoff hstart hpidbound hlaneBound ⊢
  omega

abbrev i32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

/-- The exact modular source multiplication, followed by the exact source add. -/
def blockStartBits (pid B : Nat) : BitVec 32 := i32 pid * i32 B
def offsetBits (pid B lane : Nat) : BitVec 32 := blockStartBits pid B + i32 lane

theorem i32_toNat_of_lt {n : Nat} (hn : n < 2^31) : (i32 n).toNat = n := by
  rw [BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (lt_trans hn (by norm_num))

theorem i32_toInt_of_lt {n : Nat} (hn : n < 2^31) : (i32 n).toInt = (n:Int) := by
  have hnat := i32_toNat_of_lt hn
  have hdouble : 2*(i32 n).toNat < 2^32 := by rw [hnat]; omega
  rw [BitVec.toInt_eq_toNat_of_lt hdouble, hnat]

/-- Actual i32 multiplication and addition agree with both mathematical
intermediates; the source machine operations are not replaced by a final cast. -/
theorem source_operations_exact {N B pid lane : Nat}
    (hB : LegalBlock B) (hN : N < 2^31)
    (hpid : pid < grid N B) (hlane : lane < B) :
    (i32 pid).toInt = (pid:Int) ∧ (i32 B).toInt = (B:Int) ∧
    (i32 lane).toInt = (lane:Int) ∧
    (blockStartBits pid B).toInt = ((pid*B:Nat):Int) ∧
    (offsetBits pid B lane).toInt = ((pid*B+lane:Nat):Int) ∧
    (offsetBits pid B lane).toNat = pid*B+lane := by
  have hi := source_intermediates_i32 hB hN hpid hlane
  have hoff := launched_lane_lt_i32 hB hN hpid hlane
  have hpidbound : pid < 2^31 := by exact_mod_cast hi.2.2.1.2
  have hstart : pid*B < 2^31 := by omega
  have hbits : blockStartBits pid B = i32 (pid*B) := BitVec.ofNat_mul_ofNat _ _
  have hobits : offsetBits pid B lane = i32 (pid*B+lane) := by
    unfold offsetBits
    rw [hbits]
    exact BitVec.ofNat_add_ofNat _ _
  rw [hbits, hobits]
  exact ⟨i32_toInt_of_lt hpidbound,
    i32_toInt_of_lt (legalBlock_fields hB).2.2.1,
    i32_toInt_of_lt (lt_trans hlane (legalBlock_fields hB).2.2.1),
    i32_toInt_of_lt hstart, i32_toInt_of_lt hoff, i32_toNat_of_lt hoff⟩

/-- The recorded signed-i32 comparison equals the source's mathematical mask. -/
theorem source_mask_exact {N B pid lane : Nat}
    (hB : LegalBlock B) (hN : N < 2^31)
    (hpid : pid < grid N B) (hlane : lane < B) :
    (offsetBits pid B lane).slt (i32 N) = true ↔ pid*B+lane < N := by
  have ho := (source_operations_exact hB hN hpid hlane).2.2.2.2.1
  rw [BitVec.slt_iff_toInt_lt, ho, i32_toInt_of_lt hN]
  omega

end VeriTile.Bench.TritonBenchG.SeededDropout.SourceDomain
