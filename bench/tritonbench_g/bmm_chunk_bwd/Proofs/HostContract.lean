import bench.tritonbench_g.bmm_chunk_bwd.Source
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.HostGrid

/-! Concrete source wiring and finite-grid return contract. The public direct
boundary derives this generic layout contract from admission and ordinary
allocation metadata. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostContract
open VeriTile.Triton TypedSource
noncomputable section

def parameters (h : HostCall) : RealProgram.Parameters where
  A := h.A
  Dout := h.Dout
  Db := h.Out
  Res := h.Residual
  seqlen := h.seqlen
  chunk := h.chunk
  K := h.K
  groups := h.groupCount
  aBatch := h.aStrides.batch
  aRow := h.aStrides.sequence
  aHead := h.headStride h.aStrides.head
  aColumn := h.aStrides.feature
  dBatch := h.doutStrides.batch
  dChunk := h.doutStrides.chunk
  dHead := h.headStride h.doutStrides.head
  dRow := h.doutStrides.row
  dColumn := h.doutStrides.column
  oBatch := h.outStrides.batch
  oRow := h.outStrides.sequence
  oHead := h.headStride h.outStrides.head
  oColumn := h.outStrides.feature
  rBatch := h.resStride h.residualStrides.batch
  rRow := h.resStride h.residualStrides.sequence
  rHead := h.resStride (h.headStride h.residualStrides.head)
  rColumn := h.resStride h.residualStrides.feature
  hasResidual := h.hasResidual

theorem source_eq (h : HostCall) (c : Tuning) (swapped : Bool) :
    h.source c swapped = RealProgram.source h.aStorage h.doutStorage swapped
      h.residualDType h.outputDType c.BM c.BN c.BCS (parameters h) := rfl

theorem source_real (h : HostCall) (c : Tuning) (swapped : Bool) :
    (h.source c swapped).eraseDType = RealProgram.kernel swapped c.BM c.BN c.BCS (parameters h) := by
  rw [source_eq, RealProgram.storage_erasure]

theorem tuning_positive (choice : Fin 9) :
    0 < (tuning choice).BM ∧ 0 < (tuning choice).BN ∧ 0 < (tuning choice).BCS := by
  fin_cases choice <;> decide

def grid (h : HostCall) (c : Tuning) := HostGrid.dimensions c.BM c.BN h.batch h.nchunks (parameters h)

/-- Source shape condition: square Dout chunks and all sequence chunks.
The direct wrapper validates residual and output shapes explicitly. -/
def ShapeValid (h : HostCall) : Prop :=
  0 < h.chunk ∧ 0 < h.groupCount ∧ h.nchunks = (h.seqlen + h.chunk - 1) / h.chunk

/-- Four independent source views. Positive strides are not sufficient for
nonoverlap; both within-tile and cross-program address injectivity are explicit. -/
structure Layout (h : HostCall) (c : Tuning) (alloc : FlatAlloc) : Prop where
  disjoint : alloc.Disjoint
  closed : ∀ r, r ∉ alloc.regions → alloc.extent r = 0
  outputAllocated : h.Out ∈ alloc.regions
  inputsSeparate : HostGrid.InputsSeparate (parameters h)
  outputDistinct : RealViews.OutputDistinct c.BM c.BN (parameters h)
  programWrites : HostGrid.SeparateWrites c.BM c.BN (parameters h) (HostGrid.programs (grid h c))
  bounds : ∀ q ∈ HostGrid.programs (grid h c), ∀ s,
    RealIO.Bounds (HostGrid.io Bool.false c.BM c.BN c.BCS (parameters h))
      (HostGrid.launch (grid h c) q s) alloc.extent

/-- The returned tensor descriptor is the actual output shape and strides.
For `empty_like`, these are the allocated output's strides; a supplied output
retains its own layout, which need not equal the input's. -/
structure OutputDescriptor where
  region : RegionName
  shape : List Nat
  strides : TensorStrides

def returned (h : HostCall) : OutputDescriptor := ⟨h.returnRegion, h.tensorShape, h.outStrides⟩

def run (h : HostCall) (c : Tuning) (swapped : Bool) (alloc : FlatAlloc) (s : BlockState) :
    Option (OutputDescriptor × BlockState) :=
  (HostGrid.run (alloc.flattenKernel (h.source c swapped).eraseDType.toAlgKernel)
    (grid h c) (HostGrid.programs (grid h c)) (alloc.flattenState s)).map (fun t => (returned h, t))

/-- Independent initial-memory sum at every actual active output, with
successful complete launch, returned descriptor, and complete flat frame. -/
def Correct (h : HostCall) (c : Tuning) (swapped : Bool) (alloc : FlatAlloc) (s : BlockState) : Prop :=
  ∃ t, run h c swapped alloc s = some (returned h, t) ∧
    (∀ q ∈ HostGrid.programs (grid h c), ∀ i, HostGrid.active c.BM c.BN (parameters h) q i →
      t.readMem alloc.flat (alloc.addr h.Out (HostGrid.address c.BM c.BN (parameters h) q i)) =
        HostGrid.expected c.BM c.BN (parameters h) s q i) ∧
    (∀ r a, (r ≠ alloc.flat ∨ ∀ q ∈ HostGrid.programs (grid h c), ∀ i,
      HostGrid.active c.BM c.BN (parameters h) q i →
      a ≠ alloc.addr h.Out (HostGrid.address c.BM c.BN (parameters h) q i)) →
      t.mem r a = (alloc.flattenState s).mem r a)

theorem complete (h : HostCall) (choice : Fin 9) (swapped : Bool) (alloc : FlatAlloc)
    (layout : Layout h (tuning choice) alloc) (s : BlockState) :
    Correct h (tuning choice) swapped alloc s := by
  let c := tuning choice
  have bounds : ∀ q ∈ HostGrid.programs (grid h c), ∀ s,
      RealIO.Bounds (HostGrid.io swapped c.BM c.BN c.BCS (parameters h))
        (HostGrid.launch (grid h c) q s) alloc.extent := by
    intro q hq s
    obtain ⟨hd, ha, hr, ho⟩ := layout.bounds q hq s
    exact ⟨hd, ha, hr, ho⟩
  obtain ⟨t, he, hv, hf⟩ := HostGrid.flat_correct swapped c.BM c.BN c.BCS (parameters h)
    (tuning_positive choice).2.2 layout.outputDistinct layout.inputsSeparate
    (grid h c) (HostGrid.programs (grid h c)) (HostGrid.programs_nodup _) layout.programWrites
    alloc layout.disjoint layout.closed layout.outputAllocated bounds s
  refine ⟨t, ?_, hv, hf⟩
  change (HostGrid.run (alloc.flattenKernel (h.source c swapped).eraseDType.toAlgKernel)
    (grid h c) (HostGrid.programs (grid h c)) (alloc.flattenState s)).map (fun t => (returned h, t)) = _
  rw [source_real, he]
  rfl

#specNonCircular HostGrid.expected avoiding [run]
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostContract
