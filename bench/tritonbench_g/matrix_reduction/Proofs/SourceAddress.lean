import bench.tritonbench_g.matrix_reduction.Kernels
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.Native
open VeriTile.Triton TypedSource
/-- Source storage widths, not the fp32 reduction-register width. -/
def bytes : Storage → Nat | .fp16 => 2 | .fp32 => 4
@[simp] theorem bytes_pos (d : Storage) : 0 < bytes d := by cases d <;> decide

/-- Successful source compiler admission. M is constrained by arange, and N
by the rewritten block-pointer range. The pinned block validator bounds total elements.
The external compiler contract supplies admission or a launch failure; this
is a structural frontend boundary, never a numerical assumption. -/
structure Admitted (M N : Nat) : Type where
  rows : 0 < M
  cols : 0 < N
  tensor_limit : M*N ≤ 1048576

/-- The wrapper's admitted small strides bind as i32 and are sign-extended by
make_block_ptr to i64 before multiplication by extended i32 coordinates. This
is not a wrapping i32 row*stride expression. -/
def inputPointer (base width sm sn i j : Nat) : Nat :=
  (BitVec.ofNat 64 base +
    ((BitVec.ofNat 32 i).signExtend 64 * (BitVec.ofNat 32 sm).signExtend 64 +
     (BitVec.ofNat 32 j).signExtend 64 * (BitVec.ofNat 32 sn).signExtend 64) * BitVec.ofNat 64 width).toNat
/-- stride_y is absent: the original source stores at y_ptr + arange(0,M). -/
def outputPointer (base width i : Nat) : Nat :=
  (BitVec.ofNat 64 base + (BitVec.ofNat 32 i).signExtend 64 * BitVec.ofNat 64 width).toNat

theorem extent_bounds {M N : Nat} (h : Admitted M N) : M ≤ 1048576 ∧ N ≤ 1048576 := by
  have hm := Nat.mul_le_mul_left M h.cols
  have hn := Nat.mul_le_mul_right N h.rows
  simpa using (show M*1 ≤ 1048576 ∧ 1*N ≤ 1048576 from
    ⟨hm.trans h.tensor_limit, hn.trans h.tensor_limit⟩)

theorem sign_extend_index (i : Nat) (hi : i < 2^31) :
    (BitVec.ofNat 32 i).signExtend 64 = BitVec.ofNat 64 i := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have h32 : (BitVec.ofNat 32 i).toInt = (i : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=(i : Int)) (by decide) (by omega) (by norm_num; omega)
  rw [h32]
  symm
  exact BitVec.toInt_ofInt_eq_self (w:=64) (n:=(i : Int)) (by decide) (by omega) (by norm_num; omega)

theorem linear_index_lt {M N : Nat} (i : Fin M) (j : Fin N) : i.val*N+j.val < M*N := by
  have h := Nat.mul_le_mul_right N (show i.val+1 ≤ M from i.isLt)
  nlinarith [j.isLt]

theorem input_address_exact {M N : Nat} (h : Admitted M N) (d : Storage)
    (base : Nat) (endBound : base + bytes d*(M*N) ≤ 2^64) (i : Fin M) (j : Fin N) :
    inputPointer base (bytes d) (max N 1) 1 i.val j.val = base + bytes d*(i.val*N+j.val) := by
  have hb := extent_bounds h
  have hi : i.val < 2^31 := by have := i.isLt; norm_num; omega
  have hj : j.val < 2^31 := by have := j.isLt; norm_num; omega
  have he : base+bytes d*(i.val*N+j.val) < 2^64 := by
    have hm := Nat.mul_lt_mul_of_pos_left (linear_index_lt i j) (bytes_pos d)
    omega
  unfold inputPointer
  have hsm : max N 1 < 2^31 := by have := h.cols; norm_num; omega
  rw [sign_extend_index _ hi,sign_extend_index _ hj,sign_extend_index _ hsm,
    sign_extend_index 1 (by norm_num)]
  simp only [max_eq_left (show 1 ≤ N by have := h.cols; omega),BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  simpa [Nat.mul_comm,Nat.mul_one] using Nat.mod_eq_of_lt he

theorem output_address_exact {M N : Nat} (h : Admitted M N) (d : Storage)
    (base : Nat) (endBound : base+bytes d*M ≤ 2^64) (i : Fin M) :
    outputPointer base (bytes d) i.val = base+bytes d*i.val := by
  have hb := (extent_bounds h).1
  have hi : i.val < 2^31 := by have := i.isLt; norm_num; omega
  have he : base+bytes d*i.val < 2^64 := by
    have hm := Nat.mul_lt_mul_of_pos_left i.isLt (bytes_pos d)
    omega
  unfold outputPointer
  rw [sign_extend_index _ hi,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  simpa [Nat.mul_comm] using Nat.mod_eq_of_lt he


#axiomsClean input_address_exact
#axiomsClean output_address_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixReduction.Native
