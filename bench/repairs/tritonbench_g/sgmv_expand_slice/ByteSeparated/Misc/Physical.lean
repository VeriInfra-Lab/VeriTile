import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.Native

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
open VeriTile.Triton FP.Structural Contract
open Memory
noncomputable section
open Classical

theorem output_span (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before)) (cell : Nat×Nat)
    (live : Touched (shape t s) (segments t s (lists t before)) cell) (owner address : Nat)
    (covered : (key t s cell).covers owner address) : spans.y.covers address := by
  have c:=(python_checks_iff t s spans (lists t before)).mp ok
  have cy : t.y.isContiguous=Bool.true := (c.dense t.y (by simp [tensors])).1
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,tensorKey] at lo hi
  have inside:=(a.output.window cy).active_byte (offset (shape t s) cell)
    (address-(t.y.dataPtr.toNat+4*offset (shape t s) cell))
    (output_inside ok cell (touched_complete ok cell live)) (by omega)
  have equal : t.y.dataPtr.toNat+4*offset (shape t s) cell+
      (address-(t.y.dataPtr.toNat+4*offset (shape t s) cell))=address := by omega
  rw [equal] at inside
  exact inside.1

def ReadByte (spans : Backings) (address : Nat) : Prop :=
  spans.x.covers address ∨ spans.w.covers address ∨ spans.starts.covers address ∨
    spans.lengths.covers address ∨ spans.indices.covers address

theorem separate_read (spans : Backings) (h : Separate spans) (address : Nat)
    (out : spans.y.covers address) (read : ReadByte spans address) : False := by
  rcases read with x|w|s|l|i
  · exact disjoint_byte _ _ h.1 address out x
  · exact disjoint_byte _ _ h.2.1 address out w
  · exact disjoint_byte _ _ h.2.2.1 address out s
  · exact disjoint_byte _ _ h.2.2.2.1 address out l
  · exact disjoint_byte _ _ h.2.2.2.2 address out i

/-- Every byte in each full read backing is unchanged. Physical separation
comes from the executable interval guard, never StorageImpl identity. -/
theorem caller_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before))
    (owner address : Nat) (read : ReadByte spans address) :
    sourceStores rest ops t s before owner address=before owner address := by
  have c:=(python_checks_iff t s spans (lists t before)).mp ok
  apply source_frame
  intro cell live covered
  exact separate_read spans c.separate address
    (output_span t s spans a before ok cell live owner address covered) read

/-- Any partial subset/order/payload preserves every byte outside the issued
store footprint. No numerical success claim is made after a launch failure. -/
theorem failure_complement_frame (t : Inputs) (s : Scalars) (before : Raw.State)
    (issued : List (Nat×Nat)) (payload : Nat×Nat→Raw.Word)
    (actual : ∀ cell∈issued,Touched (shape t s) (segments t s (lists t before)) cell)
    (owner address : Nat) (outside : ∀ cell,Touched (shape t s) (segments t s (lists t before)) cell →
      ¬(key t s cell).covers owner address) :
    stores Raw.ops (key t s) payload issued before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro cell member
  exact outside cell (actual cell member)

theorem failure_frame (t : Inputs) (s : Scalars) (spans : Backings) (a : Binding t spans)
    (before : Raw.State) (ok : Admitted t s spans (lists t before))
    (issued : List (Nat×Nat)) (payload : Nat×Nat→Raw.Word)
    (actual : ∀ cell∈issued,Touched (shape t s) (segments t s (lists t before)) cell)
    (owner address : Nat) (read : ReadByte spans address) :
    stores Raw.ops (key t s) payload issued before owner address=before owner address := by
  have c:=(python_checks_iff t s spans (lists t before)).mp ok
  apply failure_complement_frame t s before issued payload actual owner address
  intro cell live covered
  exact separate_read spans c.separate address
    (output_span t s spans a before ok cell live owner address covered) read

theorem empty (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State)
    (empty : (shape t s).columns=0 ∨ ∀ segment∈segments t s (lists t before),¬segment.active) :
    sourceStores rest ops t s before=before := by
  have none : cells (shape t s) (segments t s (lists t before))=[] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro cell member
    rcases ((mem_cells _ _ _).mp member).2.2 with ⟨segment,hs,active,_,_,lo,hi⟩
    rcases empty with zero|inactive
    · omega
    · exact inactive segment hs active
  simp [sourceStores,none,stores]

theorem zero_batches (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (before : Raw.State)
    (ok : Admitted t s spans (lists t before)) (zero : s.batches=0) :
    sourceStores rest ops t s before=before := by
  have c:=(python_checks_iff t s spans (lists t before)).mp ok
  have count:=(lists t before).startCount.trans c.shapes.2.2.2.1
  have starts : (lists t before).starts=[] := by simpa [zero] using count
  have noSegments : segments t s (lists t before)=[] := by simp [segments,raw,HostMetadata.rawSegments,starts]
  apply empty rest ops t s before
  right
  intro segment member
  simp [noSegments] at member

#axiomsClean output_span
#axiomsClean caller_frame
#axiomsClean failure_complement_frame
#axiomsClean failure_frame
#axiomsClean empty
#axiomsClean zero_batches
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
