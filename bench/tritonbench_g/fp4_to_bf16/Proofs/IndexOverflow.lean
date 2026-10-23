import bench.tritonbench_g.fp4_to_bf16.Proofs.FP4Bits

/-! A counterexample to the unrestricted ORIGINAL wrapper's finite-index
refinement. The original JIT takes n=2^30 as i32; its `n_elements_in * 2` is
32-bit multiplication and its output comparison is signed. These primitive
types are independently recorded by original-source TTIR and PTX in the
resumed evidence. This certificate does not execute a GPU or change Python. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.IndexOverflow
open VeriTile.Triton

def count : Nat := 1073741824
/-- Original `n_elements_out = n_elements_in * 2`, under its selected i32 ABI. -/
def outputCount : BitVec 32 := BitVec.ofNat 32 count * 2
/-- Original `mask_out = offsets_out < n_elements_out`, a signed comparison. -/
def outputMask (offset : BitVec 32) : Bool := offset.slt outputCount

theorem output_count_wraps : outputCount.toInt = -2147483648 := by decide +kernel

/-- Every output predicate is false, independently of all address/lane values.
This is stronger than a first-lane witness and covers the entire launch. -/
theorem all_output_masks_false (offset : BitVec 32) : outputMask offset = false := by
  have lower := BitVec.le_toInt offset
  norm_num at lower
  simp only [outputMask, BitVec.slt, output_count_wraps]
  simp only [decide_eq_false_iff_not]
  omega

/-- Generic masked-store primitive. Its value argument is arbitrary: no
conversion result or final-output premise is needed for the counterexample. -/
def maskedStore {α : Type} (heap : Nat → α) (address : Nat) (value : α)
    (mask : Bool) : Nat → α :=
  if mask then Function.update heap address value else heap

def laneStore {α : Type} (heap : Nat → α) (address : Nat) (value : α)
    (offset : BitVec 32) : Nat → α :=
  maskedStore heap address value (outputMask offset)

theorem no_lane_store {α : Type} (heap : Nat → α) (address : Nat) (value : α)
    (offset : BitVec 32) : laneStore heap address value offset = heap := by
  simp [laneStore, maskedStore, all_output_masks_false]

/-- Any enumeration of all lanes/programs, any addresses, and any conversion
values has no write effect. The theorem also covers any scheduling permutation. -/
theorem no_launch_stores {α : Type} (heap : Nat → α)
    (writes : List (Nat × α × BitVec 32)) :
    writes.foldl (fun memory w => laneStore memory w.1 w.2.1 w.2.2) heap = heap := by
  induction writes with
  | nil => rfl
  | cons w ws ih => simpa only [List.foldl_cons, no_lane_store] using ih

/-- Non-scalar contiguous input, a valid positive launch grid, and the actual
allocation sizes. These are exact host naturals, before device narrowing. -/
theorem host_dimensions :
    count < 2147483648 ∧ count * 2 = 2147483648 ∧
    (count + 511) / 512 = 2097152 ∧ count * 4 = 4294967296 := by decide +kernel

/-- No input address or output offset needs to overflow at this example:
all input lanes are active, all output offsets fit nonnegative int32. The bug
is solely the separately computed output-count bound. -/
theorem lane_bounds (pid lane : Nat) (hp : pid < 2097152) (hl : lane < 1024) :
    pid * 1024 + lane < 2147483648 ∧ pid * 512 + lane / 2 < count := by
  dsimp [count]
  omega

/-- One explicit disjoint, 64-bit-addressable byte layout. Actual input and
output data pointers include element storage offsets exactly once: a nonzero
input view offset and the fresh torch.empty output offset zero.
The output has 2n BF16 elements and therefore occupies 4n bytes. -/
theorem byte_storage_witness :
    let inputBase := 4294967296
    let inputOffset := 17
    let outputBase := 8589934592
    let outputOffset := 0
    let inputPointer := inputBase + inputOffset
    let outputPointer := outputBase + 2 * outputOffset
    let inputCapacity := inputOffset + count
    let outputCapacity := 2 * outputOffset + 4 * count
    inputPointer + count = inputBase + inputCapacity ∧
    outputPointer + 2 * (2 * count) = outputBase + outputCapacity ∧
    inputBase + inputCapacity < outputBase ∧
    outputBase + outputCapacity < 18446744073709551616 := by decide +kernel

/-- `torch.empty` may return the BF16 payload 0x3f80 in any output cell. For
all-zero packed input the independent target is +0. Since every store is
masked off, this permitted initial allocation disproves a universal value
contract. It asserts no concrete allocator initialization or GPU execution. -/
theorem permitted_allocation_counterexample
    (writes : List (Nat × BitVec 16 × BitVec 32)) :
    (writes.foldl (fun memory w => laneStore memory w.1 w.2.1 w.2.2)
      (fun _ => (0x3f80 : BitVec 16))) 0 ≠ Bits.expectedBits (Bits.high 0) := by
  rw [no_launch_stores]
  decide +kernel

#axiomsClean output_count_wraps
#axiomsClean all_output_masks_false
#axiomsClean no_launch_stores
#axiomsClean lane_bounds
#axiomsClean byte_storage_witness
#axiomsClean permitted_allocation_counterexample
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.IndexOverflow
