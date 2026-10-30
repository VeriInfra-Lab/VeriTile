import bench.repairs.tritonbench_g.fp4_to_bf16.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.HostSchedule

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Finite
open VeriTile.Triton VeriTile.Bench.TritonBenchG.Fp4ToBf16
open Kernels

/-- Ordinary successful storage metadata, measured in bytes. The capacity
bound is the signed host storage-size representation, not a new n cutoff.
Data pointers include storage offsets exactly once. Fresh output has offset 0.
The full allocations are disjoint; zero-capacity allocations may coincide. -/
structure Storage (n : Nat) where
  inputBase : Nat
  outputBase : Nat
  inputOffset : Nat
  inputCapacity : Nat
  outputCapacity : Nat
  inputFits : inputOffset + n ≤ inputCapacity
  outputFits : 4 * n ≤ outputCapacity
  hostCapacity : outputCapacity < 2^63
  inputAddress : inputBase + inputCapacity ≤ 2^64
  outputAddress : outputBase + outputCapacity ≤ 2^64
  separate : inputBase + inputCapacity ≤ outputBase ∨
    outputBase + outputCapacity ≤ inputBase

def Storage.inputPointer {n : Nat} (s : Storage n) := s.inputBase + s.inputOffset

theorem storage_count_bound {n : Nat} (s : Storage n) : 4*n < 9223372036854775808 := by
  have := s.outputFits
  have := s.hostCapacity
  norm_num at *
  omega

theorem padded_bounds (n pid li lo : Nat) (capacity : 4*n < 9223372036854775808)
    (hp : pid < (n+511)/512) (hi : li < 512) (ho : lo < 1024) :
    n < 9223372036854775808 ∧ 2*n < 9223372036854775808 ∧
    pid*512+li < 9223372036854775808 ∧ pid*1024+lo < 9223372036854775808 := by
  omega

theorem signed64 (n : Nat) (hn : n < 9223372036854775808) :
    (BitVec.ofNat 64 n).toInt = (n : Int) := by
  have hnat : (BitVec.ofNat 64 n).toNat = n := by
    simp only [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt, hnat]
  rw [hnat]
  norm_num
  omega

/-- Native signed-i32 program IDs and small counts are extended before use. -/
theorem widen_signed32 (n : Nat) (hn : n<2147483648) :
    (BitVec.ofNat 32 n).signExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.toInt_inj.mp
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  rw [signed64 n (by omega)]
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt, hnat]
  rw [hnat]
  norm_num
  omega

/-- Auxiliary unsigned-i32 ABI control (also used by the CPU interpreter).
Its int64 conversion is a zero extension. The native Python wrapper in the
selected Triton version chooses signed-i64 for counts at or above 2^31. -/
theorem widen_unsigned32 (n : Nat) (hn : n<4294967296) :
    (BitVec.ofNat 32 n).zeroExtend 64 = BitVec.ofNat 64 n := by
  apply BitVec.toNat_inj.mp
  change ((BitVec.ofNat 32 n).setWidth 64).toNat = _
  simp only [BitVec.toNat_setWidth, BitVec.toNat_ofNat]
  norm_num
  rw [Nat.mod_eq_of_lt hn]

/-- Actual native count specialization: constexpr for one, signed-i32 below
2^31, signed-i64 above. tl.cast accepts the constexpr case as well. -/
def nativeCount (n : Nat) : BitVec 64 :=
  if n=1 then BitVec.ofNat 64 1
  else if n<2147483648 then (BitVec.ofNat 32 n).signExtend 64
  else BitVec.ofNat 64 n

theorem native_count_refinement (n : Nat) : nativeCount n=count n := by
  unfold nativeCount count
  split_ifs with one small
  · simp only [one]
  · exact widen_signed32 n small
  · rfl

theorem native_pid_refinement (n pid : Nat)
    (gridSafe : (n+511)/512≤2147483647) (hp : pid<(n+511)/512) :
    (BitVec.ofNat 32 pid).signExtend 64=BitVec.ofNat 64 pid :=
  widen_signed32 pid (by omega)

theorem indices_exact (n pid li lo : Nat) (capacity : 4*n < 9223372036854775808)
    (hp : pid < (n+511)/512) (hi : li < 512) (ho : lo < 1024) :
    (count n).toInt = n ∧ (outputCount n).toInt = 2*n ∧
    (inputIndex pid li).toInt = pid*512+li ∧
    (outputIndex pid lo).toInt = pid*1024+lo := by
  obtain ⟨hn,hn2,hi',ho'⟩ := padded_bounds n pid li lo capacity hp hi ho
  simp only [count, outputCount, inputIndex, outputIndex,
    ← BitVec.ofNat_mul, ← BitVec.ofNat_add]
  refine ⟨signed64 n hn, ?_, ?_, ?_⟩
  · simpa [mul_comm] using signed64 (n*2) (by omega)
  · simpa only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] using signed64 _ hi'
  · simpa only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat] using signed64 _ ho'

theorem masks_exact (n pid li lo : Nat) (capacity : 4*n < 9223372036854775808)
    (hp : pid < (n+511)/512) (hi : li < 512) (ho : lo < 1024) :
    inputMask n pid li = decide (pid*512+li < n) ∧
    outputMask n pid lo = decide (pid*1024+lo < 2*n) := by
  obtain ⟨hn,hn2,hix,hox⟩ := indices_exact n pid li lo capacity hp hi ho
  simp only [inputMask, outputMask, BitVec.slt, hn, hn2, hix, hox]
  constructor <;> simp only [decide_eq_decide] <;> omega

/-- Modular pointer formation is exact on every accessed byte. Inactive padded
pointers may wrap, but the previously proved false mask suppresses all access. -/
theorem pointer_exact (base stride off : Nat) (bound : base+stride*off < 2^64) :
    (pointer base stride (BitVec.ofNat 64 off)).toNat = base+stride*off := by
  simp only [pointer, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt bound

theorem active_input_address {n : Nat} (s : Storage n) (i : Nat) (hi : i<n) :
    (pointer s.inputPointer 1 (BitVec.ofNat 64 i)).toNat = s.inputPointer+i ∧
    s.inputPointer+i+1 ≤ s.inputBase+s.inputCapacity ∧
    s.inputPointer+i+1 ≤ 2^64 := by
  have := s.inputFits
  have := s.inputAddress
  dsimp [Storage.inputPointer]
  refine ⟨?_, by omega, by omega⟩
  simpa using pointer_exact (s.inputBase+s.inputOffset) 1 i (by omega)

theorem active_output_address {n : Nat} (s : Storage n) (j : Nat) (hj : j<2*n) :
    (pointer s.outputBase 2 (BitVec.ofNat 64 j)).toNat = s.outputBase+2*j ∧
    s.outputBase+2*j+2 ≤ s.outputBase+s.outputCapacity ∧
    s.outputBase+2*j+2 ≤ 2^64 := by
  have := s.outputFits
  have := s.outputAddress
  exact ⟨pointer_exact _ _ _ (by omega), by omega, by omega⟩

theorem byte_ownership {n : Nat} (s : Storage n) (i j k : Nat)
    (hi : i<n) (hj : j<2*n) (hk : k<2) :
    s.inputPointer+i ≠ s.outputBase+2*j+k := by
  have := s.inputFits
  have := s.outputFits
  have sep := s.separate
  dsimp [Storage.inputPointer]
  rcases sep with sep | sep <;> omega

theorem distinct_output_bytes (base j k l m : Nat) (hk : k<2) (hm : m<2)
    (different : j ≠ l) : base+2*j+k ≠ base+2*l+m := by omega

theorem distinct_program_bytes (base p q j l k m : Nat) (hk : k<2) (hm : m<2)
    (hj : j/1024=p) (hl : l/1024=q) (different : p≠q) :
    base+2*j+k ≠ base+2*l+m := by
  apply distinct_output_bytes base j k l m hk hm
  omega

theorem original_counterexample_repaired :
    (outputCount 1073741824).toInt = 2147483648 ∧
    outputMask 1073741824 0 0 = Bool.true ∧
    outputMask 1073741824 2097151 1023 = Bool.true := by decide +kernel

#axiomsClean native_count_refinement
#axiomsClean native_pid_refinement
#axiomsClean indices_exact
#axiomsClean masks_exact
#axiomsClean active_input_address
#axiomsClean active_output_address
#axiomsClean byte_ownership
#axiomsClean original_counterexample_repaired
end VeriTile.Bench.Repairs.Fp4ToBf16.Finite
