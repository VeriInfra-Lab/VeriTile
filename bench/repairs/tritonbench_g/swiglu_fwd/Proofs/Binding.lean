import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Contract

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section
set_option maxHeartbeats 2000000

/-- Ordinary storage/view validity, before evaluating any source arithmetic.
Distinct storage objects may overlap. Every view reads from one shared physical
address reader per snapshot; the bounds concern tensor shapes, not outputs. -/
structure Storage (p : Prepared α input supplied entry preexisting) where
  capacity : Nat → Nat
  originalBounds : ∀ k, k < input.shape.prod → input.offset k < capacity input.allocation
  flatBounds : ∀ r q, r < p.flat.rows → q < p.flat.cols →
    p.flat.address r q < capacity p.flat.allocation
  outBounds : ∀ r q, r < p.out.rows → q < p.out.cols →
    p.out.address r q < capacity p.out.allocation
  byteBase : Nat → Nat
  allocation64 : ∀ a, byteBase a + 4 * capacity a ≤ 2^64
  device : Nat → Nat
  originalDevice : device input.allocation = input.cudaDevice
  flatDevice : device p.flat.allocation = input.cudaDevice
  outDevice : device p.out.allocation = p.out.cudaDevice
  entryRead : Nat → Nat → α
  preparedRead : Nat → Nat → α
  entryBinding : ∀ a k, k < capacity a → entry a k = entryRead (device a) (byteBase a+4*k)
  preparedBinding : ∀ a k, k < capacity a →
    p.memory a k = preparedRead (device a) (byteBase a+4*k)

/-- Pointers are the storage base plus the actual Torch storage offset. -/
def metadata (p : Prepared α input supplied entry preexisting) (s : Storage p) : LaunchMetadata :=
  ⟨p.batchShape.prod,p.ncols,p.flat.rowStride,p.out.rowStride,input.cudaDevice,
   p.out.cudaDevice,p.flat.allocation,p.out.allocation,p.flat.base,p.out.base,
   s.byteBase p.flat.allocation+4*p.flat.base,s.byteBase p.out.allocation+4*p.out.base⟩

theorem row_unique (rows cols stride r q r' q' : Nat)
    (unique : rows ≤ 1 ∨ cols ≤ stride) (hr : r < rows) (hq : q < cols)
    (hr' : r' < rows) (hq' : q' < cols)
    (same : r*stride+q = r'*stride+q') : r = r' ∧ q = q' := by
  rcases unique with singleton | width
  · have : r = 0 ∧ r' = 0 := by omega
    rcases this with ⟨rfl,rfl⟩
    simpa using same
  · have qr : q < stride := lt_of_lt_of_le hq width
    have qr' : q' < stride := lt_of_lt_of_le hq' width
    have rowsEqual : r = r' := by
      by_contra ne
      rcases lt_or_gt_of_ne ne with less | greater
      · have scaled := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt less)
        nlinarith
      · have scaled := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt greater)
        nlinarith
    subst r'
    exact ⟨rfl,by omega⟩

theorem span_bound (base rows stride width r q : Nat) (hr : r < rows) (hq : q < width) :
    base+r*stride+q < base+(rows-1)*stride+width := by
  have scaled := Nat.mul_le_mul_right stride (show r ≤ rows-1 by omega)
  omega

/-- The byte ranges of every active output cell and live input cell are
disjoint, even when distinct storage objects describe overlapping allocations. -/
def ActiveBytesSeparated (p : Prepared α input supplied entry preexisting) (s : Storage p) : Prop :=
  ∀ r q r' q', r < p.batchShape.prod → q < p.ncols →
    r' < p.batchShape.prod → q' < 2*p.ncols → ∀ bo bi, bo < 4 → bi < 4 →
      s.byteBase p.out.allocation+4*(p.out.base+r*p.out.rowStride+q)+bo ≠
        s.byteBase p.flat.allocation+4*(p.flat.base+r'*p.flat.rowStride+q')+bi

theorem active_bytes_separated (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (h : LaunchContract (metadata p s)) : ActiveBytesSeparated p s := by
  intro r q r' q' hr hq hr' hq' bo bi hbo hbi same
  have outBound := span_bound 0 p.batchShape.prod p.out.rowStride p.ncols r q hr hq
  have inBound := span_bound 0 p.batchShape.prod p.flat.rowStride (2*p.ncols) r' q' hr' hq'
  have separated := h.separate (show 0 < (metadata p s).rows from Nat.zero_lt_of_lt hr)
  dsimp [metadata,outputEnd,inputEnd] at separated
  rcases separated with before | after <;> omega

/-- Storage identity never creates a second copy of the same physical cell. -/
theorem prepared_storage_coherent (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (a b i j : Nat) (hi : i < s.capacity a) (hj : j < s.capacity b)
    (device : s.device a = s.device b)
    (address : s.byteBase a+4*i = s.byteBase b+4*j) :
    p.memory a i = p.memory b j := by
  rw [s.preparedBinding a i hi, s.preparedBinding b j hj, device, address]

theorem original_read_physical (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (i : Nat) (hi : i < input.shape.prod) :
    input.read entry i = s.entryRead input.cudaDevice
      (s.byteBase input.allocation+4*input.offset i) := by
  change entry input.allocation (input.offset i) = _
  rw [s.entryBinding _ _ (s.originalBounds i hi), s.originalDevice]

theorem prepared_reads_physical [Inhabited α]
    (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    ((p.state.mem "X" (r*p.flat.rowStride+q)).read .real =
      s.preparedRead input.cudaDevice (s.byteBase p.flat.allocation+
        4*(p.flat.base+r*p.flat.rowStride+q))) ∧
    ((p.state.mem "Y" (r*p.flat.rowStride+q)).read .real =
      s.preparedRead input.cudaDevice (s.byteBase p.flat.allocation+
        4*(p.flat.base+r*p.flat.rowStride+(p.ncols+q)))) := by
  have boundX := s.flatBounds r q (by rwa [p.flatRows]) (by rw [p.flatCols]; omega)
  have boundY := s.flatBounds r (p.ncols+q) (by rwa [p.flatRows]) (by rw [p.flatCols]; omega)
  simp only [Matrix.address,p.flatLastStride,Nat.mul_one] at boundX boundY
  constructor
  · change p.memory p.flat.allocation (p.flat.base+(r*p.flat.rowStride+q)) = _
    simpa [Nat.add_assoc,s.flatDevice] using s.preparedBinding _ _ boundX
  · change p.memory p.flat.allocation (p.flat.base+p.ncols+(r*p.flat.rowStride+q)) = _
    simpa [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm,s.flatDevice] using
      s.preparedBinding _ _ boundY

/-- Region views of the same physical address space, including actual storage
offsets. This reader can initialize raw native graph leaves without inventing
an independent memory for each storage object. -/
def physicalReads (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (region : RegionName) (offset : Nat) : α :=
  if region = "X" then s.preparedRead input.cudaDevice
    (s.byteBase p.flat.allocation+4*(p.flat.base+offset))
  else if region = "Y" then s.preparedRead input.cudaDevice
    (s.byteBase p.flat.allocation+4*(p.flat.base+p.ncols+offset))
  else s.preparedRead p.out.cudaDevice (s.byteBase p.out.allocation+4*(p.out.base+offset))

theorem physicalReads_logical [Inhabited α]
    (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    physicalReads p s "X" (r*p.flat.rowStride+q) = input.read entry (r*(2*p.ncols)+q) ∧
    physicalReads p s "Y" (r*p.flat.rowStride+q) = input.read entry (r*(2*p.ncols)+(p.ncols+q)) := by
  have physical := prepared_reads_physical p s r q hr hq
  have logical := prepared_reads p r q hr hq
  constructor
  · simpa [physicalReads,Nat.add_assoc] using physical.1.symm.trans logical.1
  · simpa [physicalReads,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using
      physical.2.symm.trans logical.2

def checkedDomain (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract (metadata p s)) : Domain p where
  positiveColumns := h.positive
  sourceChecks := hc
  sameDevice := h.sameDevice
  outputUnique := fun r q r' q' => row_unique _ _ _ r q r' q' h.unique
  capacity := s.capacity
  originalBounds := s.originalBounds
  flatBounds := by
    intro r q hr hq
    exact s.flatBounds r q (by rwa [p.flatRows]) (by rwa [p.flatCols])
  outBounds := by
    intro r q hr hq
    have shape := checks_metadata p hc
    have bound := s.outBounds r q (by rw [shape.1,p.flatRows]; exact hr)
      (by rw [shape.2.1]; exact hq)
    simpa [Matrix.address,shape.2.2] using bound
  outputOwnership := by
    intro r q r' q' hr hq hr' hq' same
    have alloc := congrArg Prod.fst same
    have address := congrArg Prod.snd same
    have outBound := span_bound p.out.base p.batchShape.prod p.out.rowStride p.ncols r q hr hq
    have inBound := span_bound p.flat.base p.batchShape.prod p.flat.rowStride (2*p.ncols) r' q' hr' hq'
    have separated := h.separate (show 0 < (metadata p s).rows from Nat.zero_lt_of_lt hr)
    dsimp [metadata,outputEnd,inputEnd] at separated
    dsimp at alloc
    simp only [Matrix.address,p.flatLastStride,Nat.mul_one] at address
    rw [alloc] at separated
    rcases separated with before | after <;> omega

/-- Only the two used storage bases need a hardware interpretation. Other
allocation identifiers are immaterial to this launch. -/
def launchBase (p : Prepared α input supplied entry preexisting) (s : Storage p) (a : Nat) : Nat :=
  if a = p.flat.allocation ∨ a = p.out.allocation then s.byteBase a else 0

def checkedFinite (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract (metadata p s))
    (nonempty : 0 < p.batchShape.prod) : FiniteAdmission p (checkedDomain p s hc h) where
  rows32 := h.rows32
  inputStride32 := h.inputStride32
  outputStride32 := h.outputStride32
  inputPadded32 := by simpa [metadata,Nat.add_assoc] using h.inputPadded32
  outputPadded32 := by simpa [metadata,Nat.add_assoc] using h.outputPadded32
  byteBase := launchBase p s
  byteBaseAligned := by
    intro a
    have aligned := h.alignment nonempty
    dsimp [metadata] at aligned
    unfold launchBase
    split
    · rename_i eq
      rcases eq with rfl | rfl <;> omega
    · rfl
  inputAllocation64 := by
    simpa [launchBase,checkedDomain,addressLimit] using s.allocation64 p.flat.allocation
  outputAllocation64 := by
    simpa [launchBase,checkedDomain,addressLimit] using s.allocation64 p.out.allocation

/-- Empty launches have no lanes or pointer accesses. Nonempty launches reuse
the full signed-i32 and pointer-i64 source correspondence. -/
def CheckedIndices (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract (metadata p s)) : Prop :=
  ∀ nonempty : 0 < p.batchShape.prod,
    IndexCorrespondence p (checkedDomain p s hc h) (checkedFinite p s hc h nonempty)

theorem checked_indices (p : Prepared α input supplied entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract (metadata p s)) : CheckedIndices p s hc h :=
  fun nonempty => index_correspondence p _ (checkedFinite p s hc h nonempty)

#axiomsClean checkedDomain
#axiomsClean checked_indices
#axiomsClean active_bytes_separated
#axiomsClean prepared_storage_coherent
#axiomsClean original_read_physical
#axiomsClean prepared_reads_physical
#axiomsClean physicalReads_logical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd
