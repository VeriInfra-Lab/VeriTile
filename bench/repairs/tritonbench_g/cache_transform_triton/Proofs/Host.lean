import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Contract

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 100000

/-- Logical byte address of a strided input view. Strides and storage offsets
are in elements; allocation identities allow metadata/cos/sin aliases. -/
def viewByteAddress {α : Type} (v : LogicalView α) (row col byte : Nat) : Nat × Nat :=
  (v.storageId, (v.offset + row * v.rowStride + col * v.colStride) * v.dtype.bytes + byte)

/-- Native payload storage, independent of any floating interpretation. -/
abbrev Byte := Fin 256
abbrev Memory := Nat → Nat → Byte

structure ByteWrite where
  (allocation offset : Nat)
  value : Byte

def performWrite (memory : Memory) (write : ByteWrite) : Memory :=
  fun allocation offset => if allocation = write.allocation ∧ offset = write.offset
    then write.value else memory allocation offset

def performWrites (memory : Memory) (writes : List ByteWrite) : Memory :=
  writes.foldl performWrite memory

/-- Library setup writes and source stores use fresh private allocations.
The theorem includes every prefix, so a failure after any completed write
preserves every caller byte. No failure-output law is a premise. -/
theorem caller_frame (memory : Memory) (writes : List ByteWrite) (caller : Nat → Prop)
    (fresh : ∀ w ∈ writes, ¬ caller w.allocation) (allocation offset : Nat)
    (hc : caller allocation) :
    performWrites memory writes allocation offset = memory allocation offset := by
  induction writes generalizing memory with
  | nil => rfl
  | cons w ws ih =>
    rw [performWrites, List.foldl_cons]
    change performWrites (performWrite memory w) ws allocation offset = _
    rw [ih (performWrite memory w) (fun x hx => fresh x (List.mem_cons_of_mem w hx))]
    have hne : allocation ≠ w.allocation := by
      intro h
      exact fresh w (List.mem_cons_self) (h ▸ hc)
    simp [performWrite, hne]

theorem failure_prefix_frame (memory : Memory) (writes : List ByteWrite)
    (caller : Nat → Prop) (fresh : ∀ w ∈ writes, ¬ caller w.allocation)
    (steps allocation offset : Nat) (hc : caller allocation) :
    performWrites memory (writes.take steps) allocation offset = memory allocation offset :=
  caller_frame memory _ caller (fun w hw => fresh w (List.mem_of_mem_take hw)) allocation offset hc

structure OutputDescriptor where
  (rows width : Nat)
  dtype : Storage
  (device allocation : Nat)
  contiguous : Bool
  requiresGrad : Bool

/-- Ordinary successful torch.empty result, parameterized by the fresh
allocation identity. The wrapper never substitutes an input view as output. -/
def allocatedOutput {α : Type} (v : LogicalView α) (mode : Mode)
    (lengths : List Nat) (allocation : Nat) : OutputDescriptor :=
  ⟨(preparedRows mode lengths).length, v.width, v.dtype, v.device, allocation, Bool.true, Bool.false⟩

theorem output_contract {α : Type} (v : LogicalView α) (mode : Mode)
    (lengths : List Nat) (allocation : Nat) :
    let out := allocatedOutput v mode lengths allocation
    out.rows = (match mode with | .prompt => lengths.sum | .decode => lengths.length) ∧
    out.width = v.width ∧ out.dtype = v.dtype ∧ out.device = v.device ∧
    out.allocation = allocation ∧ out.contiguous = Bool.true ∧ out.requiresGrad = Bool.false := by
  cases mode <;> simp [allocatedOutput, preparedRows_length]

theorem gather_empty {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (h : p.count = 0) : gather M p s = s := by
  simp [gather, blocks, h, runTiles]

/-- Zero hidden width and zero row count need no kernel, row table allocation
or positive-dimension premise. This matches the actual host count branch. -/
theorem empty_dimensions {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n width itemBytes : Nat) (h : p.count = n * (width * itemBytes))
    (empty : n = 0 ∨ width = 0) : gather M p s = s := by
  apply gather_empty M p s
  rcases empty with hn | hw
  · simp [h, hn]
  · simp [h, hw]

#axiomsClean caller_frame
#axiomsClean failure_prefix_frame
#axiomsClean output_contract
#axiomsClean empty_dimensions
end
end VeriTile.Bench.Repairs.CacheTransform
