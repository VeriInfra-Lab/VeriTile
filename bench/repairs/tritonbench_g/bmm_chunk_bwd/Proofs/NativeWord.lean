import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeControl

namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeWord
abbrev Word := BitVec 64
def enc (n : Nat) : Word := BitVec.ofNat 64 n
abbrev limit : Nat := NativeControl.signedLimit

@[simp] theorem enc_add (a b : Nat) : enc a+enc b = enc (a+b) := by
  exact BitVec.ofNat_add_ofNat a b
@[simp] theorem enc_mul (a b : Nat) : enc a*enc b = enc (a*b) := by
  exact BitVec.ofNat_mul_ofNat a b

theorem enc_nat (n : Nat) (hn : n < 2^64) : (enc n).toNat = n := by
  exact Nat.mod_eq_of_lt hn

theorem enc_msb (n : Nat) (hn : n < limit) : (enc n).msb = Bool.false := by
  apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
  rw [enc_nat n (by norm_num [NativeControl.signedLimit] at hn ⊢; omega)]
  norm_num [NativeControl.signedLimit] at hn ⊢
  omega

theorem enc_int (n : Nat) (hn : n < limit) : (enc n).toInt = (n : Int) := by
  rw [BitVec.toInt_eq_toNat_of_msb (enc_msb n hn),enc_nat n (by
    norm_num [NativeControl.signedLimit] at hn ⊢; omega)]

theorem enc_sub (a b : Nat) (hb : b < limit) (hba : b ≤ a) : enc a-enc b = enc (a-b) := by
  exact BitVec.ofNat_sub_ofNat_of_le a b (by norm_num [NativeControl.signedLimit] at hb ⊢; omega) hba

theorem enc_div (a b : Nat) (ha : a < limit) (hb : b < limit) :
    (enc a).sdiv (enc b) = enc (a/b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.sdiv_eq,enc_msb a ha,enc_msb b hb]
  change (enc a / enc b).toNat = _
  rw [BitVec.toNat_udiv,enc_nat a (by norm_num [NativeControl.signedLimit] at ha ⊢; omega),
    enc_nat b (by norm_num [NativeControl.signedLimit] at hb ⊢; omega),enc_nat (a/b)]
  exact lt_of_le_of_lt (Nat.div_le_self a b) (by norm_num [NativeControl.signedLimit] at ha ⊢; omega)

theorem enc_rem (a b : Nat) (ha : a < limit) (hb : b < limit) :
    (enc a).srem (enc b) = enc (a%b) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.srem_eq,enc_msb a ha,enc_msb b hb]
  change (enc a % enc b).toNat = _
  rw [BitVec.toNat_umod,enc_nat a (by norm_num [NativeControl.signedLimit] at ha ⊢; omega),
    enc_nat b (by norm_num [NativeControl.signedLimit] at hb ⊢; omega),enc_nat (a%b)]
  exact lt_of_le_of_lt (Nat.mod_le a b) (by norm_num [NativeControl.signedLimit] at ha ⊢; omega)

theorem enc_lt (a b : Nat) (ha : a < limit) (hb : b < limit) :
    (enc a).slt (enc b) = decide (a < b) := by
  rw [BitVec.slt_eq_decide,enc_int a ha,enc_int b hb]
  simp

def min (a b : Word) : Word := if a.slt b then a else b

theorem enc_min (a b : Nat) (ha : a < limit) (hb : b < limit) :
    min (enc a) (enc b) = enc (Min.min a b) := by
  rw [min,enc_lt a b ha hb]
  simp only [decide_eq_true_eq]
  split_ifs with h
  · rw [Nat.min_eq_left (Nat.le_of_lt h)]
  · rw [Nat.min_eq_right (Nat.le_of_not_gt h)]

def cdiv (a b : Word) : Word := (a+b-enc 1).sdiv b

theorem enc_cdiv (a b : Nat) (hbpos : 0 < b) (hb : b < limit)
    (hab : a+b-1 < limit) : cdiv (enc a) (enc b) = enc ((a+b-1)/b) := by
  rw [cdiv,enc_add,enc_sub (a+b) 1 (by decide) (by omega),enc_div _ _ hab hb]

/-- Every source pointer polynomial is represented by actual 64-bit addition
and multiplication. Overflow is allowed for inactive lanes. -/
def polynomial : NativeIndex.Polynomial → Word
  | .input n => enc n
  | .add x y => polynomial x+polynomial y
  | .mul x y => polynomial x*polynomial y

theorem polynomial_eq (e : NativeIndex.Polynomial) : polynomial e = enc (NativeIndex.ideal e) := by
  induction e with
  | input n => rfl
  | add x y hx hy => simp [polynomial,NativeIndex.ideal,hx,hy]
  | mul x y hx hy => simp [polynomial,NativeIndex.ideal,hx,hy]

#axiomsClean enc_div
#axiomsClean enc_rem
#axiomsClean enc_lt
#axiomsClean enc_cdiv
#axiomsClean polynomial_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeWord
