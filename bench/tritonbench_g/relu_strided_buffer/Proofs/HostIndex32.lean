import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedIndex32
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostPlan

/-!
The actual rank-one host plan satisfies the complete signed-i32 padded extent
domain for 0<n<2^31. This uses the existing six-shift helper as written: each
OR preserves the previous value, which suffices to lower-bound its result.
No idealized unbounded power-of-two replacement or fixed-1024 grid is used.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostIndex32
open VeriTile.Triton CappedGrid HostPlan SignedIndex32

theorem next_power_ge {n : Nat} (positive : 0 < n) : n ≤ nextPower n := by
  let x1 := (n - 1) ||| ((n - 1) >>> 1)
  let x2 := x1 ||| (x1 >>> 2)
  let x3 := x2 ||| (x2 >>> 4)
  let x4 := x3 ||| (x3 >>> 8)
  let x5 := x4 ||| (x4 >>> 16)
  let x6 := x5 ||| (x5 >>> 32)
  have h1 : n - 1 ≤ x1 := Nat.left_le_or
  have h2 : x1 ≤ x2 := Nat.left_le_or
  have h3 : x2 ≤ x3 := Nat.left_le_or
  have h4 : x3 ≤ x4 := Nat.left_le_or
  have h5 : x4 ≤ x5 := Nat.left_le_or
  have h6 : x5 ≤ x6 := Nat.left_le_or
  have bound := h1.trans (h2.trans (h3.trans (h4.trans (h5.trans h6))))
  simp only [nextPower, VeriTile.Bench.TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo,
    if_neg (by omega : n ≠ 0)]
  change n ≤ x6 + 1
  omega

theorem large_block {n : Nat} (large : 512 ≤ n) : block n = 512 := by
  exact Nat.min_eq_left (large.trans (next_power_ge (by omega)))

theorem capped_512_extent {n : Nat} (positive : 0 < n) (nBound : n < indexLimit) :
    numCTAs n 512 * tilesPerCTA n 512 * 512 ≤ indexLimit := by
  have countPositive : 0 < numTiles n 512 := cdiv_pos positive (by decide)
  have countBound : numTiles n 512 ≤ 4194304 := by
    unfold numTiles cdiv
    unfold indexLimit ReluTritonKernel.FiniteAddress.indexLimit at nBound
    omega
  by_cases uncapped : numTiles n 512 ≤ 65536
  · have c : numCTAs n 512 = numTiles n 512 := Nat.min_eq_right uncapped
    have t : tilesPerCTA n 512 = 1 := by
      rw [tilesPerCTA, c]
      exact (cdiv_eq_one_iff countPositive countPositive).mpr (by omega)
    rw [c, t]
    unfold indexLimit ReluTritonKernel.FiniteAddress.indexLimit
    omega
  · have c : numCTAs n 512 = 65536 := Nat.min_eq_left (by omega)
    have t : tilesPerCTA n 512 ≤ 64 := by
      rw [tilesPerCTA, c]
      unfold cdiv
      omega
    rw [c]
    unfold indexLimit ReluTritonKernel.FiniteAddress.indexLimit
    omega

/-- Bounds the full capped rectangular launch, including extra tail tiles.
The cap 65536 and large-input tile size 512 divide the signed limit exactly. -/
theorem padded_extent {n : Nat} (positive : 0 < n) (nBound : n < indexLimit) :
    numCTAs n (block n) * tilesPerCTA n (block n) * block n ≤ indexLimit := by
  by_cases small : n ≤ 512
  · have fits : n ≤ block n := le_min small (next_power_ge positive)
    have count : numTiles n (block n) = 1 :=
      (cdiv_eq_one_iff positive (block_pos positive)).mpr fits
    have c : numCTAs n (block n) = 1 := by simp [numCTAs, count]
    have t : tilesPerCTA n (block n) = 1 := by
      simp [tilesPerCTA, count, c, cdiv]
    rw [c, t]
    simp only [Nat.one_mul]
    exact (block_le n).trans (by decide)
  · rw [large_block (by omega : 512 ≤ n)]
    exact capped_512_extent positive nBound

theorem prepared_extent (input output : Buffer) {n : Nat}
    (sx sy : Int) (positive : 0 < n) (nBound : n < indexLimit) :
    let a := rankOneArguments input output n sx sy
    a.C * a.T * a.B ≤ indexLimit := padded_extent positive nBound

/-- Every launched iteration/lane has an exact signed-i32 source index and
mask, regardless of whether the lane is active. -/
theorem prepared_index_and_mask (input output : Buffer) {n pid j lane : Nat}
    (sx sy : Int) (positive : 0 < n) (nBound : n < indexLimit)
    (launched : pid < (rankOneArguments input output n sx sy).C)
    (iteration : j < (rankOneArguments input output n sx sy).T)
    (inTile : lane < (rankOneArguments input output n sx sy).B) :
    let a := rankOneArguments input output n sx sy
    (index32 a.B a.C pid j lane).toInt = (logicalIndex a.B a.C pid j lane : Nat) ∧
    mask32 n a.B a.C pid j lane = decide (logicalIndex a.B a.C pid j lane < n) ∧
    (index32 a.B a.C pid j lane).signExtend 64 =
      BitVec.ofNat 64 (logicalIndex a.B a.C pid j lane) := by
  have extent := prepared_extent input output sx sy positive nBound
  exact ⟨(arithmetic_exact extent launched iteration inTile).2.2.2.1,
    mask_exact nBound extent launched iteration inTile,
    index_sign_extend extent launched iteration inTile⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.HostIndex32
