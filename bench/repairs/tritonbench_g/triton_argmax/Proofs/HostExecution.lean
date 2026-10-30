import bench.repairs.tritonbench_g.triton_argmax.Proofs.Host

/-! Composition from original logical view through packing, initialized
scratch, full grids and the actual returned allocation. Byte frames are derived
from ordinary fresh-allocation disjointness and primitive store frames. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.HostExecution
open Source Memory Host

 theorem word_point_apart (a b : Allocation) (apart : a.Disjoint b) (width index : Nat)
    (inside : width*(index+1) ≤ a.bytes) (point : Address) (pointInside : b.contains point) :
    Disjoint (address a width index) width point 1 := by
  obtain ⟨owner,lower,upper⟩ := pointInside
  rcases apart with owners | before | after
  · left
    dsimp [address]
    intro same
    exact owners (same.trans owner)
  · right; left
    dsimp [address] at *
    nlinarith
  · right; right
    dsimp [address] at *
    omega

theorem init_input {M : Type} {d : MidFormat} (mem : Primitives M) (c : GlobalConfig)
    (p : GlobalMemory.Buffers) (layout : GlobalMemory.Layout d c p) (s : M)
    (i : Nat) (bound : i < c.size) :
    GlobalMemory.input mem p (initializeIndices mem p c.midBlock s) i=GlobalMemory.input mem p s i := by
  unfold GlobalMemory.input initializeIndices
  apply read_frame
  intro j member
  apply subword_disjoint _ _ layout.inputIndex
  · exact (Nat.mul_le_mul_left 8 (Nat.succ_le_iff.mpr (List.mem_range.mp member))).trans layout.indexCapacity
  · exact (Nat.mul_le_mul_left 4 (Nat.succ_le_iff.mpr bound)).trans layout.inputCapacity

theorem global_source_congr {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (a b : Nat → Word) (same : ∀ i,i < c.size → a i=b i) :
    Source.globalIndex ops c a=Source.globalIndex ops c b := by
  have first := GlobalMemory.source_congr ops c a b same
  have selected := GlobalMemory.second_selected_congr ops c (stage1Value ops c a) (stage1Value ops c b)
    (fun i _ => (first i).1)
  dsimp only [Source.globalIndex]
  rw [selected]
  split
  · exact (first _).2
  · rfl

structure GlobalSetup {M : Type} (mem : Primitives M) (before : M) (original : Tensor)
    (d : MidFormat) where
  packed : Packed mem before original
  c : GlobalConfig
  p : GlobalMemory.Buffers
  output : Output
  inputBuffer : p.input=packed.tensor.dataWindow
  outputBuffer : p.output=output.returned.storage
  size : c.size=original.numel
  layout : GlobalMemory.Layout d c p
  freshMid : p.midValue.Disjoint original.storage
  freshIndex : p.midIndex.Disjoint original.storage
  freshOutput : p.output.Disjoint original.storage

def globalRun {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : GlobalSetup mem before original d) : M :=
  GlobalMemory.run mem ops setup.c setup.p
    (initializeIndices mem setup.p setup.c.midBlock setup.packed.state)

theorem global_inputs {M : Type} {d : MidFormat} (mem : Primitives M)
    (before : M) (original : Tensor) (setup : GlobalSetup mem before original d)
    (i : Nat) (bound : i < setup.c.size) :
    GlobalMemory.input mem setup.p (initializeIndices mem setup.p setup.c.midBlock setup.packed.state) i=
      logicalWord mem before original i := by
  rw [init_input mem setup.c setup.p setup.layout setup.packed.state i bound]
  unfold GlobalMemory.input
  rw [setup.inputBuffer]
  exact setup.packed.coordinates i (setup.size ▸ bound)

theorem global_returned {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : GlobalSetup mem before original d) :
    mem.read 8 (globalRun mem ops before original setup)
      (address setup.output.returned.storage 8 setup.output.returned.storageOffset)=
      BitVec.ofNat 64 (Source.globalIndex ops setup.c (logicalWord mem before original)) := by
  have offset : setup.output.returned.storageOffset=0 := setup.output.sameOffset.trans setup.output.zeroOffset
  rw [offset,←setup.outputBuffer]
  unfold globalRun
  rw [GlobalMemory.global_readback mem ops setup.c setup.p setup.layout _
    (initialized mem setup.p setup.c.midBlock setup.packed.state)]
  rw [global_source_congr ops setup.c _ _ (global_inputs mem before original setup)]

theorem global_caller_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : GlobalSetup mem before original d)
    (point : Address) (inside : original.storage.contains point) :
    mem.byte (globalRun mem ops before original setup) point=mem.byte before point := by
  have indexApart : ∀ i,i < setup.c.midBlock → Disjoint (address setup.p.midIndex 8 i) 8 point 1 := by
    intro i hi
    exact word_point_apart _ _ setup.freshIndex 8 i
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans setup.layout.indexCapacity) point inside
  unfold globalRun
  rw [GlobalMemory.global_bytes_frame]
  · unfold initializeIndices
    rw [bytes_frame]
    · exact setup.packed.callerFrame point inside
    · intro i hi
      exact indexApart i (List.mem_range.mp hi)
  · intro i hi
    exact word_point_apart _ _ setup.freshMid _ i
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans setup.layout.midCapacity) point inside
  · intro i hi
    exact indexApart i (hi.trans_le setup.layout.midFits)
  · exact word_point_apart _ _ setup.freshOutput 8 0 (by simpa using setup.layout.outputCapacity) point inside

structure DimSetup {M : Type} (mem : Primitives M) (before : M) (original : Tensor) where
  packed : Packed mem before original
  c : DimConfig
  p : DimMemory.Buffers
  output : Output
  inputBuffer : p.input=packed.tensor.dataWindow
  outputBuffer : p.output=output.returned.storage
  size : c.outer*c.extent*c.inner=original.numel
  layout : DimMemory.Layout c p
  freshOutput : p.output.Disjoint original.storage

def dimensionRun {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : DimSetup mem before original) : M :=
  DimMemory.run mem ops setup.c setup.p setup.packed.state

theorem dimension_inputs {M : Type} (mem : Primitives M) (before : M) (original : Tensor)
    (setup : DimSetup mem before original) (i : Nat) (bound : i < setup.c.outer*setup.c.extent*setup.c.inner) :
    DimMemory.input mem setup.p setup.packed.state i=logicalWord mem before original i := by
  unfold DimMemory.input
  rw [setup.inputBuffer]
  exact setup.packed.coordinates i (setup.size ▸ bound)

theorem dimension_returned {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : DimSetup mem before original)
    (row inner : Nat) (hr : row < setup.c.outer) (hi : inner < setup.c.inner) :
    mem.read 8 (dimensionRun mem ops before original setup)
      (address setup.output.returned.storage 8 (setup.output.returned.storageOffset+dimOutputOffset setup.c row inner))=
      BitVec.ofNat 64 (dimIndex ops setup.c (logicalWord mem before original) row inner) := by
  have offset : setup.output.returned.storageOffset=0 := setup.output.sameOffset.trans setup.output.zeroOffset
  rw [offset,Nat.zero_add,←setup.outputBuffer]
  unfold dimensionRun
  rw [DimMemory.dimension_readback mem ops setup.c setup.p setup.layout setup.packed.state row inner hr hi,
    DimMemory.source_congr ops setup.c _ _ (dimension_inputs mem before original setup) row inner hi]

theorem dimension_caller_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : DimSetup mem before original)
    (point : Address) (inside : original.storage.contains point) :
    mem.byte (dimensionRun mem ops before original setup) point=mem.byte before point := by
  unfold dimensionRun
  rw [DimMemory.dimension_bytes_frame mem ops setup.c setup.p setup.layout setup.packed.state point]
  · exact setup.packed.callerFrame point inside
  · intro row inner hr hi
    have bound := Native.dim_output_bound _ _ _ _ hr hi
    exact word_point_apart _ _ setup.freshOutput 8 _
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr bound)).trans setup.layout.outputCapacity) point inside

/-- Ordinary allocator freshness for all pre-existing caller allocations,
not only the input backing storage. Packing may allocate zero or more fresh
work buffers, and every library write is confined to those buffers. -/
def FreshFor (fresh : Allocation) (callers : List Allocation) : Prop :=
  ∀ caller,caller∈callers → fresh.Disjoint caller

theorem disjoint_not_contains (a b : Allocation) (apart : a.Disjoint b)
    (point : Address) (inside : b.contains point) : ¬a.contains point := by
  intro also
  obtain ⟨ob,lb,ub⟩ := inside
  obtain ⟨oa,la,ua⟩ := also
  rcases apart with owners | before | after
  · exact owners (oa.symm.trans ob)
  · dsimp only at before
    omega
  · dsimp only at after
    omega

theorem packed_frame {M : Type} (mem : Primitives M) (before : M) (original : Tensor)
    (packed : Packed mem before original) (callers : List Allocation)
    (fresh : ∀ a,a∈packed.libraryWrites → FreshFor a callers)
    (caller : Allocation) (member : caller∈callers) (point : Address) (inside : caller.contains point) :
    mem.byte packed.state point=mem.byte before point := by
  apply packed.libraryFrame
  intro a ha
  exact disjoint_not_contains a caller (fresh a ha caller member) point inside

theorem global_all_caller_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : GlobalSetup mem before original d)
    (callers : List Allocation)
    (packFresh : ∀ a,a∈setup.packed.libraryWrites → FreshFor a callers)
    (midFresh : FreshFor setup.p.midValue callers)
    (indexFresh : FreshFor setup.p.midIndex callers)
    (outputFresh : FreshFor setup.p.output callers)
    (caller : Allocation) (member : caller∈callers) (point : Address) (inside : caller.contains point) :
    mem.byte (globalRun mem ops before original setup) point=mem.byte before point := by
  have indexApart : ∀ i,i < setup.c.midBlock → Disjoint (address setup.p.midIndex 8 i) 8 point 1 := by
    intro i hi
    exact word_point_apart _ _ (indexFresh caller member) 8 i
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans setup.layout.indexCapacity) point inside
  unfold globalRun
  rw [GlobalMemory.global_bytes_frame]
  · unfold initializeIndices
    rw [bytes_frame]
    · exact packed_frame mem before original setup.packed callers packFresh caller member point inside
    · intro i hi
      exact indexApart i (List.mem_range.mp hi)
  · intro i hi
    exact word_point_apart _ _ (midFresh caller member) _ i
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hi)).trans setup.layout.midCapacity) point inside
  · intro i hi
    exact indexApart i (hi.trans_le setup.layout.midFits)
  · exact word_point_apart _ _ (outputFresh caller member) 8 0
      (by simpa using setup.layout.outputCapacity) point inside

theorem dimension_all_caller_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (before : M) (original : Tensor) (setup : DimSetup mem before original)
    (callers : List Allocation)
    (packFresh : ∀ a,a∈setup.packed.libraryWrites → FreshFor a callers)
    (outputFresh : FreshFor setup.p.output callers)
    (caller : Allocation) (member : caller∈callers) (point : Address) (inside : caller.contains point) :
    mem.byte (dimensionRun mem ops before original setup) point=mem.byte before point := by
  unfold dimensionRun
  rw [DimMemory.dimension_bytes_frame mem ops setup.c setup.p setup.layout setup.packed.state point]
  · exact packed_frame mem before original setup.packed callers packFresh caller member point inside
  · intro row inner hr hi
    have bound := Native.dim_output_bound _ _ _ _ hr hi
    exact word_point_apart _ _ (outputFresh caller member) 8 _
      ((Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr bound)).trans setup.layout.outputCapacity) point inside

#axiomsClean packed_frame
#axiomsClean global_all_caller_frame
#axiomsClean dimension_all_caller_frame
#axiomsClean init_input
#axiomsClean global_inputs
#axiomsClean global_returned
#axiomsClean global_caller_frame
#axiomsClean dimension_inputs
#axiomsClean dimension_returned
#axiomsClean dimension_caller_frame
end VeriTile.Bench.Repairs.TritonArgmax.HostExecution
