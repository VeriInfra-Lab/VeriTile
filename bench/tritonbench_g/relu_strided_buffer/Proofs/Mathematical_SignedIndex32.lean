import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_SignedByteAddress
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_CappedGrid

/-!
The source's actual pid + j*num_ctas, tile_id*B and +lane expressions at width
32. Bounds cover every launched lane, including inactive tail tiles/lanes.
The explicit domain is the complete padded extent C*T*B <= 2^31, with n<2^31
for signed mask comparison. No active-only overflow assumption is used.

These are word-expression theorems. Native instruction/operand selection is
the ordinary compiler contract, and arbitrary larger logical sizes are not
claimed. A separate host corollary derives this domain from the source plan.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedIndex32
open VeriTile.Triton CappedGrid SignedView SignedByteAddress

abbrev indexLimit : Nat := ReluTritonKernel.FiniteAddress.indexLimit

def iterationProduct32 (C j : Nat) : BitVec 32 :=
  BitVec.ofNat 32 j * BitVec.ofNat 32 C

def tileId32 (C pid j : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid + iterationProduct32 C j

def tileStart32 (B C pid j : Nat) : BitVec 32 :=
  tileId32 C pid j * BitVec.ofNat 32 B

def index32 (B C pid j lane : Nat) : BitVec 32 :=
  tileStart32 B C pid j + BitVec.ofNat 32 lane

def mask32 (n B C pid j lane : Nat) : Bool :=
  (index32 B C pid j lane).slt (BitVec.ofNat 32 n)

theorem tile_bounds {C T pid j : Nat} (launched : pid < C) (iteration : j < T) :
    pid + j * C < C * T := by
  have product := Nat.mul_le_mul_right C (Nat.succ_le_of_lt iteration)
  rw [Nat.succ_mul] at product
  rw [Nat.mul_comm C T]
  omega

theorem lane_padded_bound {B C T pid j lane : Nat}
    (launched : pid < C) (iteration : j < T) (inTile : lane < B) :
    logicalIndex B C pid j lane < C * T * B := by
  have tile := tile_bounds launched iteration
  have product := Nat.mul_le_mul_right B (Nat.succ_le_of_lt tile)
  rw [Nat.succ_mul] at product
  unfold logicalIndex
  omega

/-- Every source arithmetic result is below the signed-positive limit, even
when its lane will subsequently be masked off. -/
theorem intermediate_bounds {B C T pid j lane : Nat}
    (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B) :
    j * C < indexLimit ∧ pid + j * C < indexLimit ∧
    (pid + j * C) * B < indexLimit ∧ logicalIndex B C pid j lane < indexLimit := by
  have laneBound := Nat.lt_of_lt_of_le (lane_padded_bound launched iteration inTile) extent
  have positive : 0 < B := by omega
  have tileLe : pid + j * C ≤ (pid + j * C) * B := by
    exact Nat.le_mul_of_pos_right _ positive
  unfold logicalIndex at laneBound ⊢
  omega

theorem iteration_product_word (C j : Nat) :
    iterationProduct32 C j = BitVec.ofNat 32 (j * C) := by
  simp only [iterationProduct32, BitVec.ofNat_mul_ofNat]

theorem tile_id_word (C pid j : Nat) :
    tileId32 C pid j = BitVec.ofNat 32 (pid + j * C) := by
  simp only [tileId32, iteration_product_word, BitVec.ofNat_add_ofNat]

theorem tile_start_word (B C pid j : Nat) :
    tileStart32 B C pid j = BitVec.ofNat 32 ((pid + j * C) * B) := by
  simp only [tileStart32, tile_id_word, BitVec.ofNat_mul_ofNat]

theorem index_word (B C pid j lane : Nat) :
    index32 B C pid j lane = BitVec.ofNat 32 (logicalIndex B C pid j lane) := by
  simp only [index32, tile_start_word, logicalIndex, BitVec.ofNat_add_ofNat]

theorem arithmetic_exact {B C T pid j lane : Nat}
    (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B) :
    (iterationProduct32 C j).toInt = (j * C : Nat) ∧
    (tileId32 C pid j).toInt = (pid + j * C : Nat) ∧
    (tileStart32 B C pid j).toInt = ((pid + j * C) * B : Nat) ∧
    (index32 B C pid j lane).toInt = (logicalIndex B C pid j lane : Nat) ∧
    (index32 B C pid j lane).toNat = logicalIndex B C pid j lane := by
  obtain ⟨productBound, tileBound, startBound, laneBound⟩ :=
    intermediate_bounds extent launched iteration inTile
  rw [iteration_product_word, tile_id_word, tile_start_word, index_word]
  refine ⟨ReluTritonKernel.FiniteAddress.nat32_signed _ productBound,
    ReluTritonKernel.FiniteAddress.nat32_signed _ tileBound,
    ReluTritonKernel.FiniteAddress.nat32_signed _ startBound,
    ReluTritonKernel.FiniteAddress.nat32_signed _ laneBound, ?_⟩
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
  unfold indexLimit ReluTritonKernel.FiniteAddress.indexLimit at laneBound
  omega

theorem mask_exact {n B C T pid j lane : Nat}
    (nBound : n < indexLimit) (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B) :
    mask32 n B C pid j lane = decide (logicalIndex B C pid j lane < n) := by
  unfold mask32
  rw [BitVec.slt_eq_decide, (arithmetic_exact extent launched iteration inTile).2.2.2.1,
    ReluTritonKernel.FiniteAddress.nat32_signed n nBound]
  simp only [Int.ofNat_lt]

theorem inactive_mask_false {n B C T pid j lane : Nat}
    (nBound : n < indexLimit) (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B)
    (inactive : n ≤ logicalIndex B C pid j lane) :
    mask32 n B C pid j lane = Bool.false := by
  rw [mask_exact nBound extent launched iteration inTile]
  simp only [Nat.not_lt.mpr inactive, decide_false]

theorem index_sign_extend {B C T pid j lane : Nat}
    (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B) :
    (index32 B C pid j lane).signExtend 64 =
      BitVec.ofNat 64 (logicalIndex B C pid j lane) := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide),
    (arithmetic_exact extent launched iteration inTile).2.2.2.1]
  symm
  have bound := (intermediate_bounds extent launched iteration inTile).2.2.2
  exact BitVec.toInt_ofInt_eq_self (w := 64) (n := (logicalIndex B C pid j lane : Nat))
    (by decide) (by omega) (by
      unfold indexLimit ReluTritonKernel.FiniteAddress.indexLimit at bound
      norm_num
      omega)

/-- Composes the proved source 32-bit index/sign extension with the signed
64-bit address bridge. Only genuinely active bytes are dereferenced. -/
theorem active_byte_address_exact (allocationBase : Nat) (v : View)
    {n capacity B C T pid j lane byte : Nat}
    (valid : v.Valid n capacity) (allocation : AllocationDomain allocationBase capacity)
    (extent : C * T * B ≤ indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B)
    (active : logicalIndex B C pid j lane < n) (inElement : byte < 4) :
    (dataPtr64 allocationBase v + (index32 B C pid j lane).signExtend 64 *
      BitVec.ofInt 64 v.stride * 4 + BitVec.ofNat 64 byte).toNat =
      allocationBase + 4 * v.address (logicalIndex B C pid j lane) + byte := by
  rw [index_sign_extend extent launched iteration inTile]
  exact active_byte_exact allocationBase v valid allocation active inElement

/-- The one-tile branch is iteration zero of the same index formula. -/
theorem one_tile_word (B C pid lane : Nat) :
    index32 B C pid 0 lane =
      BitVec.ofNat 32 pid * BitVec.ofNat 32 B + BitVec.ofNat 32 lane := by
  simp only [index32, tileStart32, tileId32, iterationProduct32,
    BitVec.zero_mul, BitVec.add_zero]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedIndex32
