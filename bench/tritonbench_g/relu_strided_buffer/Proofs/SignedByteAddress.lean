import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedView
import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteAddress

/-!
Local FP32 address bridge for StridedBuffer: allocation base + 4 * view.base,
then logical index * signed stride * 4. Negative strides are accumulated as
integers before conversion to an unsigned physical address. The corresponding
64-bit expression is ordinary modular addition and multiplication; its final
value is exact for active cells in a valid, non-wrapping allocation domain.

The logical index is supplied as a Nat and encoded at width 64. This module
does not prove that native finite-width tile_id/offset/lane calculations produce
that encoding, or certify native instruction selection. Ordinary allocation,
pointer provenance, byte representation and runtime contracts stay external.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedByteAddress
open VeriTile.Triton SignedView

abbrev AllocationDomain (allocationBase capacity : Nat) : Prop :=
  ReluTritonKernel.FiniteAddress.ByteDomain allocationBase 4 capacity

def dataPtrInt (allocationBase : Nat) (v : View) : Int :=
  (allocationBase : Int) + 4 * v.base

def elementPtrInt (allocationBase : Nat) (v : View) (i : Nat) : Int :=
  dataPtrInt allocationBase v + (i : Int) * v.stride * 4

def bytePtrInt (allocationBase : Nat) (v : View) (i byte : Nat) : Int :=
  elementPtrInt allocationBase v i + (byte : Int)

def dataPtr64 (allocationBase : Nat) (v : View) : BitVec 64 :=
  BitVec.ofNat 64 allocationBase + 4 * BitVec.ofInt 64 v.base

def elementPtr64 (allocationBase : Nat) (v : View) (i : Nat) : BitVec 64 :=
  dataPtr64 allocationBase v +
    BitVec.ofNat 64 i * BitVec.ofInt 64 v.stride * 4

def bytePtr64 (allocationBase : Nat) (v : View) (i byte : Nat) : BitVec 64 :=
  elementPtr64 allocationBase v i + BitVec.ofNat 64 byte

/-- The source's complete signed expression equals the physical element cell;
no signed addend is individually truncated to Nat. -/
theorem element_int_exact (allocationBase : Nat) (v : View) {i : Nat}
    (validAddress : 0 ≤ v.signedAddress i) :
    elementPtrInt allocationBase v i =
      ((allocationBase + 4 * v.address i : Nat) : Int) := by
  simp only [elementPtrInt, dataPtrInt, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
  rw [address_cast v validAddress]
  simp only [View.signedAddress]
  ring

theorem byte_int_exact (allocationBase : Nat) (v : View) {i : Nat}
    (validAddress : 0 ≤ v.signedAddress i) (byte : Nat) :
    bytePtrInt allocationBase v i byte =
      ((allocationBase + 4 * v.address i + byte : Nat) : Int) := by
  simp only [bytePtrInt, element_int_exact allocationBase v validAddress, Nat.cast_add]

/-- Ring operations on encoded signed values agree modulo 2^64. Intermediate
negative values need no unsigned interpretation. -/
theorem element_word_encoding (allocationBase : Nat) (v : View) (i : Nat) :
    elementPtr64 allocationBase v i = BitVec.ofInt 64 (elementPtrInt allocationBase v i) := by
  simp only [elementPtr64, dataPtr64, elementPtrInt, dataPtrInt,
    BitVec.ofInt_add, BitVec.ofInt_mul, BitVec.ofInt_natCast, BitVec.ofInt_ofNat]
  rfl

theorem byte_word_encoding (allocationBase : Nat) (v : View) (i byte : Nat) :
    bytePtr64 allocationBase v i byte = BitVec.ofInt 64 (bytePtrInt allocationBase v i byte) := by
  simp only [bytePtr64, bytePtrInt, element_word_encoding, BitVec.ofInt_add,
    BitVec.ofInt_natCast]

/-- All four bytes of every active FP32 cell lie inside the allocation and the
unsigned 64-bit address space. This uses the final signed physical index. -/
theorem active_byte_span (allocationBase : Nat) (v : View) {n capacity i byte : Nat}
    (valid : v.Valid n capacity) (domain : AllocationDomain allocationBase capacity)
    (active : i < n) (inElement : byte < 4) :
    allocationBase ≤ allocationBase + 4 * v.address i + byte ∧
    allocationBase + 4 * v.address i + byte < allocationBase + 4 * capacity ∧
    allocationBase + 4 * v.address i + byte < 2^64 := by
  simpa only [Nat.mul_comm] using
    ReluTritonKernel.FiniteAddress.byte_span allocationBase 4 capacity (v.address i) byte
      domain (address_lt v valid active) inElement

theorem active_byte_exact (allocationBase : Nat) (v : View) {n capacity i byte : Nat}
    (valid : v.Valid n capacity) (domain : AllocationDomain allocationBase capacity)
    (active : i < n) (inElement : byte < 4) :
    (bytePtr64 allocationBase v i byte).toNat = allocationBase + 4 * v.address i + byte := by
  rw [byte_word_encoding, byte_int_exact allocationBase v (valid i active).1,
    BitVec.ofInt_natCast, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (active_byte_span allocationBase v valid domain active inElement).2.2

theorem active_element_exact (allocationBase : Nat) (v : View) {n capacity i : Nat}
    (valid : v.Valid n capacity) (domain : AllocationDomain allocationBase capacity)
    (active : i < n) :
    (elementPtr64 allocationBase v i).toNat = allocationBase + 4 * v.address i := by
  simpa only [bytePtr64, BitVec.add_zero, Nat.add_zero] using
    active_byte_exact allocationBase v valid domain active (byte := 0) (by decide)

/-- Nonempty valid views also justify the source host data_ptr offset itself. -/
theorem data_ptr_exact (allocationBase : Nat) (v : View) {n capacity : Nat}
    (valid : v.Valid n capacity) (domain : AllocationDomain allocationBase capacity)
    (nonempty : 0 < n) :
    (dataPtr64 allocationBase v).toNat = allocationBase + 4 * v.base.toNat := by
  simpa only [elementPtr64, BitVec.zero_mul, BitVec.add_zero,
    View.address, View.signedAddress, Nat.cast_zero, zero_mul, add_zero] using
    active_element_exact allocationBase v valid domain (i := 0) nonempty

/-- Interface for already supplied machine operands. Exact logical index-word
formation remains an explicit premise rather than a native-index claim. -/
theorem supplied_operands_exact (allocationBase : Nat) (v : View) {n capacity i : Nat}
    (valid : v.Valid n capacity) (domain : AllocationDomain allocationBase capacity)
    (active : i < n) (pointer index stride : BitVec 64)
    (pointerEncoding : pointer = dataPtr64 allocationBase v)
    (indexEncoding : index = BitVec.ofNat 64 i)
    (strideEncoding : stride = BitVec.ofInt 64 v.stride) :
    (pointer + index * stride * 4).toNat = allocationBase + 4 * v.address i := by
  subst pointer index stride
  exact active_element_exact allocationBase v valid domain active

/-- Distinct active logical cells and byte positions cannot share a physical
byte when the output stride is nonzero, including negative output strides. -/
theorem active_bytes_injective (allocationBase : Nat) (v : View) {n capacity i j a b : Nat}
    (valid : v.Valid n capacity) (strideNonzero : v.stride ≠ 0)
    (activeI : i < n) (activeJ : j < n) (byteA : a < 4) (byteB : b < 4)
    (same : allocationBase + 4 * v.address i + a = allocationBase + 4 * v.address j + b) :
    i = j ∧ a = b := by
  have same' : allocationBase + v.address i * 4 + a =
      allocationBase + v.address j * 4 + b := by simpa only [Nat.mul_comm] using same
  obtain ⟨cells, bytes⟩ := ReluTritonKernel.FiniteAddress.byte_coordinates_injective
    allocationBase 4 (v.address i) (v.address j) a b byteA byteB same'
  have indices := address_injective v valid strideNonzero
    (a₁ := ⟨i, activeI⟩) (a₂ := ⟨j, activeJ⟩) cells
  exact ⟨congrArg Fin.val indices, bytes⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedByteAddress
