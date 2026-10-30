import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.WordHost

/-! Finite source access correspondence for the same prepared host arguments.
Allocation/provenance and byte representation are ordinary memory contracts.
No execution result, address result or output value is supplied as a premise. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton FP.Structural HostPlan SignedView SignedGrid

/-- Native FP32 allocation metadata. Region names denote allocation identity:
equal regions share one base; different regions have disjoint byte ranges. -/
structure NativeBuffers (input output : Buffer) (inputCapacity outputCapacity : Nat) where
  inputBase : Nat
  outputBase : Nat
  inputAlignment : inputBase % 4 = 0
  outputAlignment : outputBase % 4 = 0
  inputAllocation : SignedByteAddress.AllocationDomain inputBase inputCapacity
  outputAllocation : SignedByteAddress.AllocationDomain outputBase outputCapacity
  sameAllocation : input.region = output.region → inputBase = outputBase
  separateAllocations : input.region ≠ output.region →
    inputBase + 4 * inputCapacity ≤ outputBase ∨
      outputBase + 4 * outputCapacity ≤ inputBase

/-- Complete index, mask and active-byte relation. The index is the actual
`(pid + j*num_ctas)*tile_size0 + lane` expression of both source branches.
All lanes, including masked ones, have exact i32 index/mask computations. -/
def NativeAccess (input output : Buffer) (n : Nat) (sx sy : Int)
    {inputCapacity outputCapacity : Nat}
    (b : NativeBuffers input output inputCapacity outputCapacity) : Prop :=
  let a := rankOneArguments input output n sx sy
  (SignedByteAddress.dataPtr64 b.inputBase a.input).toNat =
    b.inputBase + 4 * input.base.toNat ∧
  (SignedByteAddress.dataPtr64 b.outputBase a.output).toNat =
    b.outputBase + 4 * output.base.toNat ∧
  ∀ pid j lane, pid < a.C → j < a.T → lane < a.B →
    let logical := CappedGrid.logicalIndex a.B a.C pid j lane
    let index := SignedIndex32.index32 a.B a.C pid j lane
    index.toInt = (logical : Int) ∧
    SignedIndex32.mask32 n a.B a.C pid j lane = decide (logical < n) ∧
    (logical < n → ∀ byte, byte < 4 →
      (SignedByteAddress.dataPtr64 b.inputBase a.input + index.signExtend 64 *
        BitVec.ofInt 64 sx * 4 + BitVec.ofNat 64 byte).toNat =
        b.inputBase + 4 * a.input.address logical + byte ∧
      (SignedByteAddress.dataPtr64 b.outputBase a.output + index.signExtend 64 *
        BitVec.ofInt 64 sy * 4 + BitVec.ofNat 64 byte).toNat =
        b.outputBase + 4 * a.output.address logical + byte ∧
      b.inputBase ≤ b.inputBase + 4 * a.input.address logical + byte ∧
      b.inputBase + 4 * a.input.address logical + byte < b.inputBase + 4 * inputCapacity ∧
      b.inputBase + 4 * a.input.address logical + byte < 2^64 ∧
      b.outputBase ≤ b.outputBase + 4 * a.output.address logical + byte ∧
      b.outputBase + 4 * a.output.address logical + byte < b.outputBase + 4 * outputCapacity ∧
      b.outputBase + 4 * a.output.address logical + byte < 2^64)

theorem native_access (input output : Buffer) {n inputCapacity outputCapacity : Nat}
    (positive : 0 < n) (bounded : n < SignedIndex32.indexLimit) (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity)
    (b : NativeBuffers input output inputCapacity outputCapacity) :
    NativeAccess input output n sx sy b := by
  refine ⟨SignedByteAddress.data_ptr_exact b.inputBase (input.view sx) hin b.inputAllocation positive,
    SignedByteAddress.data_ptr_exact b.outputBase (output.view sy) hout b.outputAllocation positive, ?_⟩
  intro pid j lane hp hj hl
  obtain ⟨index, mask, pointers⟩ := WordHost.prepared_accesses input output positive bounded sx sy
    hin hout b.inputBase b.outputBase b.inputAllocation b.outputAllocation hp hj hl
  refine ⟨index, mask, ?_⟩
  intro active byte byteBound
  obtain ⟨inputPointer, outputPointer⟩ := pointers active byte byteBound
  obtain ⟨ilo, ihi, iword⟩ := SignedByteAddress.active_byte_span b.inputBase (input.view sx)
    hin b.inputAllocation active byteBound
  obtain ⟨olo, ohi, oword⟩ := SignedByteAddress.active_byte_span b.outputBase (output.view sy)
    hout b.outputAllocation active byteBound
  exact ⟨inputPointer, outputPointer, ilo, ihi, iword, olo, ohi, oword⟩

/-- The mapped source address operation and a machine byte pointer agree.
This uses the actual `dynamicAddress` operation in `SignedViews.perTile` and
its original offset register; it is not a second unrelated address program. -/
theorem source_pointer_lowering (allocationBase : Nat) (v : View)
    {n capacity B C T pid j lane byte : Nat} (valid : v.Valid n capacity)
    (allocation : SignedByteAddress.AllocationDomain allocationBase capacity)
    (padded : C * T * B ≤ SignedIndex32.indexLimit)
    (launched : pid < C) (iteration : j < T) (inTile : lane < B)
    (active : CappedGrid.logicalIndex B C pid j lane < n) (inByte : byte < 4)
    (s : BlockState)
    (offset : s.regs .nat [] "offset0" =
      some (Tile.scalar ((pid + j * C) * B))) :
    evalOp (SourceSignedBranch.dynamicAddress v B) s =
      some (SourceSignedBranch.addressValue v ((pid + j * C) * B) B) ∧
    (SignedByteAddress.dataPtr64 allocationBase v +
      (SignedIndex32.index32 B C pid j lane).signExtend 64 *
        BitVec.ofInt 64 v.stride * 4 + BitVec.ofNat 64 byte).toNat =
      allocationBase + 4 *
        (SourceSignedBranch.addressValue v ((pid + j * C) * B) B).data
          (⟨lane, inTile⟩, PUnit.unit) + byte := by
  refine ⟨SourceSignedBranch.dynamicAddress_eval v B _ s offset, ?_⟩
  exact SignedIndex32.active_byte_address_exact allocationBase v valid allocation
    padded launched iteration inTile active inByte

/-- Logical nonaliasing and ordinary allocation provenance imply separation
of all four physical bytes of every input/output pair. -/
theorem input_output_bytes_disjoint (input output : Buffer)
    {n inputCapacity outputCapacity i j u v : Nat} (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity)
    (separate : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (b : NativeBuffers input output inputCapacity outputCapacity)
    (hi : i < n) (hj : j < n) (hu : u < 4) (hv : v < 4) :
    b.inputBase + 4 * (input.view sx).address i + u ≠
      b.outputBase + 4 * (output.view sy).address j + v := by
  by_cases same : input.region = output.region
  · rw [b.sameAllocation same]
    intro collision
    have coordinates := ReluTritonKernel.FiniteAddress.byte_coordinates_injective
      b.outputBase 4 ((input.view sx).address i) ((output.view sy).address j) u v hu hv
      (by simpa only [Nat.mul_comm] using collision)
    rcases separate i hi j hj with regions | cells
    · exact regions same
    · exact cells coordinates.1
  · simpa only [Nat.mul_comm] using ReluTritonKernel.FiniteAddress.separate_byte_ranges
      b.inputBase 4 inputCapacity b.outputBase 4 outputCapacity
      ((input.view sx).address i) ((output.view sy).address j) u v
      (by simpa only [Nat.mul_comm] using b.separateAllocations same)
      (address_lt _ hin hi) (address_lt _ hout hj) hu hv

/-- Every concrete byte pointer in the finite statement is the physical
image of the actual source address operation at its computed offset register. -/
def SourceAddressLowering (input output : Buffer) (n : Nat) (sx sy : Int)
    {inputCapacity outputCapacity : Nat}
    (b : NativeBuffers input output inputCapacity outputCapacity) : Prop :=
  let a := rankOneArguments input output n sx sy
  ∀ pid j lane, pid < a.C → j < a.T → lane < a.B →
    CappedGrid.logicalIndex a.B a.C pid j lane < n → ∀ byte, byte < 4 →
    ∀ s : BlockState, s.regs .nat [] "offset0" =
      some (Tile.scalar ((pid + j * a.C) * a.B)) →
    evalOp (SourceSignedBranch.dynamicAddress a.input a.B) s =
      some (SourceSignedBranch.addressValue a.input ((pid + j * a.C) * a.B) a.B) ∧
    evalOp (SourceSignedBranch.dynamicAddress a.output a.B) s =
      some (SourceSignedBranch.addressValue a.output ((pid + j * a.C) * a.B) a.B) ∧
    (SignedByteAddress.dataPtr64 b.inputBase a.input +
      (SignedIndex32.index32 a.B a.C pid j lane).signExtend 64 *
        BitVec.ofInt 64 sx * 4 + BitVec.ofNat 64 byte).toNat =
      b.inputBase + 4 * a.input.address (CappedGrid.logicalIndex a.B a.C pid j lane) + byte ∧
    (SignedByteAddress.dataPtr64 b.outputBase a.output +
      (SignedIndex32.index32 a.B a.C pid j lane).signExtend 64 *
        BitVec.ofInt 64 sy * 4 + BitVec.ofNat 64 byte).toNat =
      b.outputBase + 4 * a.output.address (CappedGrid.logicalIndex a.B a.C pid j lane) + byte

theorem source_address_lowering (input output : Buffer) {n inputCapacity outputCapacity : Nat}
    (positive : 0 < n) (bounded : n < SignedIndex32.indexLimit) (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity)
    (b : NativeBuffers input output inputCapacity outputCapacity) :
    SourceAddressLowering input output n sx sy b := by
  intro pid j lane hp hj hl active byte hb s offset
  have extent := HostIndex32.prepared_extent input output sx sy positive bounded
  obtain ⟨load, inputBytes⟩ := source_pointer_lowering b.inputBase (input.view sx)
    hin b.inputAllocation extent hp hj hl active hb s offset
  obtain ⟨store, outputBytes⟩ := source_pointer_lowering b.outputBase (output.view sy)
    hout b.outputAllocation extent hp hj hl active hb s offset
  exact ⟨load, store, inputBytes, outputBytes⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
