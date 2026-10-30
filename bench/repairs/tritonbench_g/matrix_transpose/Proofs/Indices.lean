import bench.repairs.tritonbench_g.matrix_transpose.Proofs.Contract

namespace VeriTile.Bench.Repairs.MatrixTranspose.Indices

/-- Sum of the source's two signed-int32 stride products. -/
def offset32 (row stride col : Nat) : BitVec 32 :=
  BitVec.ofNat 32 row * BitVec.ofNat 32 stride + BitVec.ofNat 32 col * 1

/-- Actual left-associated source expression: base + row*stride + col*1.
The two element offsets are separately sign-extended and scaled for FP16. -/
def pointer64 (base : BitVec 64) (row stride col : Nat) : BitVec 64 :=
  base + (BitVec.ofNat 32 row * BitVec.ofNat 32 stride).signExtend 64 * 2 +
    (BitVec.ofNat 32 col * 1).signExtend 64 * 2

theorem flat_bound (m d row col : Nat) (hr : row < m) (hc : col < d) :
    row*d+col < m*d := by
  have := Nat.mul_le_mul_right d (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at this
  omega

theorem dimensions_bound (m d : Nat) (h : InputContract m d) :
    m ≤ 2^20 ∧ d ≤ 2^20 := by
  have hm := Nat.le_mul_of_pos_right m h.positive.2
  have hd := Nat.le_mul_of_pos_left d h.positive.1
  exact ⟨hm.trans h.extent, hd.trans h.extent⟩

theorem offset32_exact (row stride col : Nat) (h : row*stride+col < 2^31) :
    (offset32 row stride col).toInt = (row*stride+col : Nat) := by
  have word : offset32 row stride col = BitVec.ofNat 32 (row*stride+col) := by
    simp [offset32, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
  rw [word]
  exact BitVec.toInt_ofInt_eq_self (w := 32) (n := (row*stride+col : Nat))
    (by decide) (by omega) (by norm_num; omega)

/-- All arange values, runtime strides, both products and final offsets fit
signed int32. No mask removes invalid lanes: the entire tile is in bounds. -/
theorem all_indices (m d : Nat) (h : InputContract m d)
    (row : Fin m) (col : Fin d) :
    m < 2^31 ∧ d < 2^31 ∧
    row.val*d+col.val < m*d ∧ col.val*m+row.val < m*d ∧
    (offset32 row.val d col.val).toInt = (row.val*d+col.val : Nat) ∧
    (offset32 col.val m row.val).toInt = (col.val*m+row.val : Nat) := by
  have hi := flat_bound m d row.val col.val row.isLt col.isLt
  have ho : col.val*m+row.val < m*d := by
    simpa [Nat.mul_comm] using flat_bound d m col.val row.val col.isLt row.isLt
  have hd := dimensions_bound m d h
  have he := h.extent
  exact ⟨by omega, by omega, hi, ho,
    offset32_exact _ _ _ (by omega), offset32_exact _ _ _ (by omega)⟩

private theorem extend_exact (n : Nat) (h : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = BitVec.ofNat 64 n := by
    apply BitVec.eq_of_toInt_eq
    rw [BitVec.toInt_signExtend_of_le (by decide)]
    have small : (BitVec.ofNat 32 n).toInt = (n : Int) :=
      BitVec.toInt_ofInt_eq_self (w := 32) (n := (n : Int))
        (by decide) (by omega) (by norm_num; omega)
    rw [small]
    symm
    exact BitVec.toInt_ofInt_eq_self (w := 64) (n := (n : Int))
      (by decide) (by omega) (by norm_num; omega)

/-- Both actual pointer additions equal their natural byte addresses. The
ordinary allocator's address-space bound rules out uint64 wrapping. -/
theorem pointer_exact (base : BitVec 64) (row stride col : Nat)
    (indexBound : row*stride+col < 2^31)
    (allocationBound : base.toNat + 2*(row*stride+col)+1 < 2^64) :
    (pointer64 base row stride col).toNat = base.toNat+2*(row*stride+col) := by
  have productBound : row*stride < 2^31 := by omega
  have colBound : col < 2^31 := by omega
  unfold pointer64
  rw [show (1 : BitVec 32) = BitVec.ofNat 32 1 from rfl]
  simp only [BitVec.ofNat_mul_ofNat, Nat.mul_one]
  rw [extend_exact _ productBound, extend_exact _ colBound]
  have base_word : base = BitVec.ofNat 64 base.toNat := by simp
  nth_rw 1 [base_word]
  rw [show (2 : BitVec 64) = BitVec.ofNat 64 2 from rfl,
    BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.ofNat_mul_ofNat,
    BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  omega

/-- Both bytes lie in the ordinary allocated FP16 window and are naturally
aligned. This is derived for every unmasked lane, including the last one. -/
theorem allocation_access (base : BitVec 64) (m d : Nat) (h : InputContract m d)
    (row : Fin m) (col : Fin d)
    (allocationBound : base.toNat + 2*(m*d) ≤ 2^64)
    (alignment : base.toNat % 2 = 0) :
    (pointer64 base row.val d col.val).toNat = base.toNat+2*(row.val*d+col.val) ∧
    (pointer64 base row.val d col.val).toNat % 2 = 0 ∧
    base.toNat ≤ (pointer64 base row.val d col.val).toNat ∧
    (pointer64 base row.val d col.val).toNat+1 < base.toNat+2*(m*d) := by
  have bound := flat_bound m d row.val col.val row.isLt col.isLt
  have extent := h.extent
  have pointer := pointer_exact base row.val d col.val (by omega) (by omega)
  rw [pointer]
  exact ⟨rfl, by omega, by omega, by omega⟩

#axiomsClean all_indices
#axiomsClean pointer_exact
#axiomsClean allocation_access
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixTranspose.Indices
