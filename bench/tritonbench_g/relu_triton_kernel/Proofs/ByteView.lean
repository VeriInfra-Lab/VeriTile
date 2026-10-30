import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteStores

/-! Representation relation between mathematical bytes and complete tagged
logical cells. Distinct logical names and explicit physical interval separation
are subtheorem premises, not claims about empty_like or source runtime guards. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory
open VeriTile.Triton FP.Structural IntegerWords FiniteAddress GatedGrid

def LogicalView (encoding : Encoding) (bytes : Memory) (inputBase outputBase N : Nat)
    (x y : RegionName) (logical : GatedGrid.Memory (BitVec 32)) : Prop :=
  (∀ index, index < N → logical x index = decodeCell encoding (loadInput encoding bytes inputBase index)) ∧
  (∀ index, index < N → logical y index = .mk .fp32 (load32 bytes outputBase index))

/-- A logical view can be constructed from arbitrary initial bytes. Other
logical cells/registers remain arbitrary. This is not a native allocation. -/
def stateFromBytes (encoding : Encoding) (bytes : Memory) (inputBase outputBase N : Nat)
    (x y : RegionName) (initial : State (BitVec 32)) : State (BitVec 32) :=
  { initial with mem := fun r o =>
      if r = x ∧ o < N then decodeCell encoding (loadInput encoding bytes inputBase o)
      else if r = y ∧ o < N then .mk .fp32 (load32 bytes outputBase o) else initial.mem r o }

theorem state_from_bytes_view (encoding : Encoding) (bytes : Memory)
    (inputBase outputBase N : Nat) (x y : RegionName) (initial : State (BitVec 32))
    (distinct : x ≠ y) :
    LogicalView encoding bytes inputBase outputBase N x y
      (stateFromBytes encoding bytes inputBase outputBase N x y initial).mem := by
  constructor
  · intro index active
    simp [stateFromBytes, active]
  · intro index active
    simp [stateFromBytes, active, Ne.symm distinct]

theorem input_words_of_view (encoding : Encoding) (bytes : Memory) (inputBase outputBase N : Nat)
    (x y : RegionName) (logical : GatedGrid.Memory (BitVec 32))
    (view : LogicalView encoding bytes inputBase outputBase N x y logical) :
    InputWords encoding (loadInput encoding bytes inputBase) x N logical := view.1

theorem byte_grid_output (M : Algebra (BitVec 32)) (encoding : Encoding)
    (inputBase outputBase N index : Nat) (initial : Memory) (active : index < N) :
    load32 (byteGrid M encoding inputBase outputBase N initial) outputBase index =
      if index < blockSize then values M encoding inputBase initial index else load32 initial outputBase index := by
  rw [byte_grid_stores]
  by_cases first : index < blockSize
  · rw [if_pos first]
    exact prefix_load32 _ _ _ _ _ (Nat.lt_min.mpr ⟨active, first⟩)
  · rw [if_neg first]
    exact prefix_load32_tail _ _ _ _ _ (le_trans (Nat.min_le_right _ _) (by omega))

theorem byte_grid_input (M : Algebra (BitVec 32)) (encoding : Encoding)
    (inputBase outputBase N index : Nat) (initial : Memory)
    (separate : Separate encoding inputBase outputBase N) (active : index < N) :
    loadInput encoding (byteGrid M encoding inputBase outputBase N initial) inputBase index =
      loadInput encoding initial inputBase index := by
  rw [byte_grid_stores]
  exact prefix_input_preserved encoding _ inputBase outputBase N _ index initial separate
    (Nat.min_le_left _ _) active

theorem byte_grid_frame (M : Algebra (BitVec 32)) (encoding : Encoding)
    (inputBase outputBase N address : Nat) (initial : Memory)
    (outside : address < outputBase ∨ outputBase + min N blockSize * 4 ≤ address) :
    byteGrid M encoding inputBase outputBase N initial address = initial address := by
  rw [byte_grid_stores]
  exact prefix_frame _ _ _ _ _ outside

theorem word_spec_view (M : Algebra (BitVec 32)) (encoding : Encoding)
    (inputBase outputBase N : Nat) (x y : RegionName) (initial : Memory)
    (logical : GatedGrid.Memory (BitVec 32)) (distinct : x ≠ y)
    (separate : Separate encoding inputBase outputBase N)
    (view : LogicalView encoding initial inputBase outputBase N x y logical) :
    LogicalView encoding (byteGrid M encoding inputBase outputBase N initial) inputBase outputBase N x y
      (wordSpec M encoding (loadInput encoding initial inputBase) y N blockSize logical) := by
  constructor
  · intro index active
    rw [wordSpec_frame M encoding _ y N blockSize logical x index (Or.inl distinct),
      view.1 index active, byte_grid_input M encoding inputBase outputBase N index initial separate active]
  · intro index active
    rw [byte_grid_output M encoding inputBase outputBase N index initial active]
    by_cases first : index < blockSize
    · rw [if_pos first, wordSpec_written_fp32 M encoding _ y N blockSize logical index first active]
      rfl
    · rw [if_neg first, wordSpec_frame M encoding _ y N blockSize logical y index
        (Or.inr (Or.inl (by omega)))]
      exact view.2 index active

/-- Concrete byte readback at the proved64-bit address, including each last
element byte. Numeric bounds do not establish pointer provenance. -/
theorem finite_output_byte (M : Algebra (BitVec 32)) (encoding : Encoding)
    (inputBase outputBase N index byte : Nat) (initial : Memory)
    (indexDomain : IndexDomain N) (outputDomain : ByteDomain outputBase 4 N)
    (first : index < blockSize) (active : index < N) (within : byte < 4) :
    byteGrid M encoding inputBase outputBase N initial
        (bytePointer64 outputBase 4 0 index byte).toNat =
      splitByte (values M encoding inputBase initial index) byte := by
  have launched : 0 < gridCount N blockSize := positive_grid_count N blockSize (by omega) (by decide)
  have actual : offset 0 index < N := by simpa [offset] using active
  rw [byte_pointer_exact outputBase 4 N 0 index byte indexDomain outputDomain launched first actual within]
  simp only [offset, Nat.zero_mul, Nat.zero_add]
  rw [byte_grid_stores]
  exact prefix_byte _ _ _ _ _ _ (Nat.lt_min.mpr ⟨active, first⟩) within

theorem finite_input_word (encoding : Encoding) (inputBase N pid lane : Nat) (initial : Memory)
    (indexDomain : IndexDomain N) (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize)
    (active : offset pid lane < N) :
    (loadAt64 (elementBytes encoding) initial (pointer64 inputBase (elementBytes encoding) pid lane)).cast
        (element_bytes_width encoding) = loadInput encoding initial inputBase (offset pid lane) := by
  rw [finite_pointer_load (elementBytes encoding) inputBase N pid lane initial indexDomain inputDomain
    (element_bytes_positive encoding) launched inBlock active]
  rfl

#specNonCircular LogicalView avoiding [TypedValue.source, runGrid, FP.Structural.exec,
  byteProgram, bytePrefix, byteGrid, wordSpec]
#axiomsClean state_from_bytes_view
#axiomsClean input_words_of_view
#axiomsClean byte_grid_output
#axiomsClean byte_grid_input
#axiomsClean byte_grid_frame
#axiomsClean word_spec_view
#axiomsClean finite_output_byte
#axiomsClean finite_input_word
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory
