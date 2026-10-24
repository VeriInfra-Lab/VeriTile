import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteAddress

/-! N is a Python constexpr, not an annotated signed-i32 argument. At exactly
2^31 the frontend promotes the mask comparison to unsigned i32; all launched
offsets still fit signed i32. This valid endpoint is included. Above it the
first additional block overflows the signed source multiply before the gate. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteIndices
open VeriTile.Triton GatedGrid FiniteAddress

def Domain (N : Nat) : Prop := N ≤ indexLimit

theorem padded_extent (N : Nat) (domain : Domain N) :
    gridCount N blockSize * blockSize ≤ indexLimit := by
  unfold Domain indexLimit at domain
  unfold gridCount blockSize indexLimit
  omega

theorem lane_bound (N pid lane : Nat) (domain : Domain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    offset pid lane < indexLimit := by
  have h := padded_extent N domain
  unfold offset blockSize at *
  omega

theorem offset_word (N pid lane : Nat) (domain : Domain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    (offset32 pid lane).toNat = offset pid lane ∧
    (offset32 pid lane).toInt = (offset pid lane : Int) := by
  have bound := lane_bound N pid lane domain launched inBlock
  rw [offset32_word]
  refine ⟨?_, nat32_signed _ bound⟩
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
  unfold indexLimit at bound
  omega

theorem offset_sign_extend (N pid lane : Nat) (domain : Domain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    (offset32 pid lane).signExtend 64 = BitVec.ofNat 64 (offset pid lane) := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),
    (offset_word N pid lane domain launched inBlock).2]
  symm
  have bound := lane_bound N pid lane domain launched inBlock
  exact BitVec.toInt_ofInt_eq_self (w := 64) (n := (offset pid lane : Nat))
    (by decide) (by omega) (by unfold indexLimit at bound; norm_num; omega)

def sourceMask (N pid lane : Nat) : Bool :=
  if N < indexLimit then FiniteAddress.mask32 N pid lane
  else (offset32 pid lane).ult (BitVec.ofNat 32 N)

theorem mask_exact (N pid lane : Nat) (domain : Domain N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize) :
    sourceMask N pid lane = decide (offset pid lane < N) := by
  unfold sourceMask
  split
  · exact FiniteAddress.mask_exact N pid lane ‹N < indexLimit› launched inBlock
  · rw [BitVec.ult_eq_decide, (offset_word N pid lane domain launched inBlock).1,
      BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
    unfold Domain indexLimit at domain
    omega

theorem byte_pointer_exact (base bytes N pid lane byte : Nat)
    (indexDomain : Domain N) (domain : ByteDomain base bytes N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize)
    (active : offset pid lane < N) (inElement : byte < bytes) :
    (bytePointer64 base bytes pid lane byte).toNat = base + offset pid lane * bytes + byte := by
  have bound := (byte_span base bytes N (offset pid lane) byte domain active inElement).2.2
  unfold bytePointer64 pointer64
  rw [offset_sign_extend N pid lane indexDomain launched inBlock,
    BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.ofNat_add_ofNat,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound]

theorem boundary_last_lane :
    (offset32 (2^21-1) 1023).toInt = (2^31-1 : Int) := by decide +kernel

theorem next_block_overflows_before_gate :
    (start32 (2^21)).toInt = -(2^31 : Int) ∧
    sourceMask (2^31+1) (2^21) 0 = Bool.true := by decide +kernel

#axiomsClean padded_extent
#axiomsClean lane_bound
#axiomsClean offset_word
#axiomsClean offset_sign_extend
#axiomsClean mask_exact
#axiomsClean byte_pointer_exact
#axiomsClean boundary_last_lane
#axiomsClean next_block_overflows_before_gate
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteIndices
