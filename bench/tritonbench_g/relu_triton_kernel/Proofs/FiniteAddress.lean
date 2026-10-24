import bench.tritonbench_g.relu_triton_kernel.Proofs.WordGrid

/-! Fixed-1024 ReLU index arithmetic on an explicit nonnegative signed-i32
domain. All launched lanes, including inactive tail lanes, satisfy the
integer bounds. These are expression-level refinement lemmas; they do not
assert native dispatch, compiler refinement, launch validity or success. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress
open VeriTile.Triton GatedGrid

def blockSize : Nat := 1024
def indexLimit : Nat := 2^31
def IndexDomain (N : Nat) : Prop := N < indexLimit
def offset (pid lane : Nat) : Nat := pid * blockSize + lane
def start32 (pid : Nat) : BitVec 32 := BitVec.ofNat 32 pid * BitVec.ofNat 32 blockSize
def offset32 (pid lane : Nat) : BitVec 32 := start32 pid + BitVec.ofNat 32 lane
def mask32 (N pid lane : Nat) : Bool := (offset32 pid lane).slt (BitVec.ofNat 32 N)
def stores32 (pid : Nat) : Bool := BitVec.ofNat 32 pid == BitVec.ofNat 32 0

/-- Since 1024 divides 2^31, ceiling to a complete tile does not cross the
signed-positive limit even when N is 2^31-1. -/
theorem padded_extent_le (N : Nat) (domain : IndexDomain N) :
    gridCount N blockSize * blockSize ≤ indexLimit := by
  unfold IndexDomain indexLimit at domain
  unfold gridCount blockSize indexLimit
  omega

theorem lane_bound (N pid lane : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    offset pid lane < indexLimit := by
  have h := padded_extent_le N domain
  unfold offset blockSize at *
  omega

theorem pid_bound (N pid : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) : pid < indexLimit := by
  have h := lane_bound N pid 0 domain launched (by decide)
  unfold offset blockSize at h
  omega

theorem nat32_signed (n : Nat) (bound : n < indexLimit) :
    (BitVec.ofNat 32 n).toInt = (n : Int) := by
  exact BitVec.toInt_ofInt_eq_self (w := 32) (n := (n : Int))
    (by decide) (by omega) (by unfold indexLimit at bound; norm_num; omega)

theorem offset32_word (pid lane : Nat) :
    offset32 pid lane = BitVec.ofNat 32 (offset pid lane) := by
  simp only [offset32, start32, offset, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

/-- Both the multiply and the subsequent add fit, not merely the final
masked addresses. Native debug/sanitizer outcomes remain outside this proof. -/
theorem lane_arithmetic (N pid lane : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    (start32 pid).toInt = (pid * blockSize : Nat) ∧
    (offset32 pid lane).toInt = (offset pid lane : Nat) ∧
    (offset32 pid lane).toNat = offset pid lane := by
  have bound := lane_bound N pid lane domain launched inBlock
  have productBound : pid * blockSize < indexLimit := by unfold offset at bound; omega
  refine ⟨?_, ?_, ?_⟩
  · rw [start32, BitVec.ofNat_mul_ofNat]
    exact nat32_signed _ productBound
  · rw [offset32_word]
    exact nat32_signed _ bound
  · rw [offset32_word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
    unfold indexLimit at bound
    omega

theorem mask_exact (N pid lane : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    mask32 N pid lane = decide (offset pid lane < N) := by
  unfold mask32
  rw [BitVec.slt_eq_decide, (lane_arithmetic N pid lane domain launched inBlock).2.1,
    nat32_signed N domain]
  simp only [Int.ofNat_lt]

theorem stores_exact (N pid : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) :
    stores32 pid = decide (pid = 0) := by
  unfold stores32
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq, ← BitVec.toInt_inj, nat32_signed pid (pid_bound N pid domain launched),
    decide_eq_true_eq]
  norm_num

/-- Sign extension preserves the nonnegative bounded element index before
the separately modelled 64-bit byte-pointer arithmetic. -/
theorem offset_sign_extend (N pid lane : Nat) (domain : IndexDomain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    (offset32 pid lane).signExtend 64 = BitVec.ofNat 64 (offset pid lane) := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),
    (lane_arithmetic N pid lane domain launched inBlock).2.1]
  symm
  have bound := lane_bound N pid lane domain launched inBlock
  exact BitVec.toInt_ofInt_eq_self (w := 64) (n := (offset pid lane : Nat))
    (by decide) (by omega) (by unfold indexLimit at bound; norm_num; omega)

#specNonCircular mask32 avoiding [TypedValue.source, runGrid, FP.Structural.exec]
#axiomsClean padded_extent_le
#axiomsClean lane_bound
#axiomsClean pid_bound
#axiomsClean nat32_signed
#axiomsClean offset32_word
#axiomsClean lane_arithmetic
#axiomsClean mask_exact
#axiomsClean stores_exact
#axiomsClean offset_sign_extend
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress
