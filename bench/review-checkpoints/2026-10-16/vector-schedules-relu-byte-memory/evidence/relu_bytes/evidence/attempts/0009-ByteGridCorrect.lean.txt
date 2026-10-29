import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteView

/-! Mathematical byte-memory refinement of the nine-word ReLU grid.
The arbitrary32-bit output callbacks remain explicit. The complete tagged
view, distinct logical names and disjoint byte intervals are subtheorem
premises; no native allocator, compiler or provenance theorem is claimed.
No assumptions constrain the arbitrary initial output tail or other bytes. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteGridCorrect
open VeriTile.Triton FP.Structural GatedGrid TypedValue IntegerWords FiniteAddress ByteMemory
noncomputable section

specification relu_byte_grid_correctness
    (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (encoding : Encoding) (inputBase outputBase N : Nat) (x y : RegionName)
    (bytes : ByteMemory.Memory) (initial : State (BitVec 32))
    (view : LogicalView encoding bytes inputBase outputBase N x y initial.mem)
    (distinct : x ≠ y) (separate : Separate encoding inputBase outputBase N)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (outputDomain : ByteDomain outputBase 4 N) :
    ∃ final,
      runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize)
        N blockSize initial = some final ∧
      final.mem = wordSpec M encoding (loadInput encoding bytes inputBase) y N blockSize initial.mem ∧
      LogicalView encoding (byteGrid M encoding inputBase outputBase N bytes)
        inputBase outputBase N x y final.mem ∧
      byteGrid M encoding inputBase outputBase N bytes = byteSpec M encoding inputBase outputBase N bytes ∧
      (∀ address, address < outputBase ∨ outputBase + min N blockSize * 4 ≤ address →
        byteGrid M encoding inputBase outputBase N bytes address = bytes address) ∧
      (∀ index, index < blockSize → index < N → ∀ byte, byte < 4 →
        byteGrid M encoding inputBase outputBase N bytes
          (bytePointer64 outputBase 4 0 index byte).toNat =
            splitByte (ByteMemory.values M encoding inputBase bytes index) byte) ∧
      (∀ pid, pid < gridCount N blockSize → ∀ lane, lane < blockSize → offset pid lane < N →
        (loadAt64 (elementBytes encoding) bytes (pointer64 inputBase (elementBytes encoding) pid lane)).cast
          (element_bytes_width encoding) = loadInput encoding bytes inputBase (offset pid lane)) ∧
      AddressGridCorrect.AddressCertificate encoding N inputBase outputBase := by
  obtain ⟨final, he, hm, _, address⟩ := AddressGridCorrect.relu_word_grid_finite_address_correctness
    M cmp16 cmp32 encoding (loadInput encoding bytes inputBase) x y N inputBase outputBase initial
    (input_words_of_view encoding bytes inputBase outputBase N x y initial.mem view)
    indexDomain inputDomain outputDomain
  refine ⟨final, he, hm, ?_, byte_grid_correct M encoding inputBase outputBase N bytes,
    fun address outside => byte_grid_frame M encoding inputBase outputBase N address bytes outside,
    fun index first active byte within => finite_output_byte M encoding inputBase outputBase N index byte bytes
      indexDomain outputDomain first active within,
    fun pid launched lane inBlock active => finite_input_word encoding inputBase N pid lane bytes
      indexDomain inputDomain launched inBlock active, address⟩
  rw [hm]
  exact word_spec_view M encoding inputBase outputBase N x y bytes initial.mem distinct separate view

/-- Arbitrary initial bytes admit the required logical representation when
the chosen logical names are distinct. This removes the view premise without
assuming native representation or changing any byte of the initial memory. -/
theorem constructed_byte_grid
    (M : Algebra (BitVec 32)) (cmp16 cmp32 : BitVec 32 → BitVec 32 → Bool)
    (encoding : Encoding) (inputBase outputBase N : Nat) (x y : RegionName)
    (bytes : ByteMemory.Memory) (initial : State (BitVec 32))
    (distinct : x ≠ y) (separate : Separate encoding inputBase outputBase N)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (outputDomain : ByteDomain outputBase 4 N) :
    ∃ final,
      runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize) N blockSize
        (stateFromBytes encoding bytes inputBase outputBase N x y initial) = some final ∧
      LogicalView encoding (byteGrid M encoding inputBase outputBase N bytes)
        inputBase outputBase N x y final.mem := by
  obtain ⟨final, he, _, hv, _⟩ := relu_byte_grid_correctness M cmp16 cmp32 encoding inputBase outputBase N x y bytes
    (stateFromBytes encoding bytes inputBase outputBase N x y initial)
    (state_from_bytes_view encoding bytes inputBase outputBase N x y initial distinct)
    distinct separate indexDomain inputDomain outputDomain
  exact ⟨final, he, hv⟩

#axiomsClean relu_byte_grid_correctness
#axiomsClean constructed_byte_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteGridCorrect
