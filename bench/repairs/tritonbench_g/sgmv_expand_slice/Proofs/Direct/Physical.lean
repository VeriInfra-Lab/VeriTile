import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Native
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Direct.Bounds

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.NativeCall
open VeriTile.Triton FP.Structural Contract
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

 theorem output_span (ok : Admitted t s (lists a before)) (cell : Nat×Nat)
    (live : Touched (shape t s) (segments t s (lists a before)) cell) (owner address : Nat)
    (covered : (key a s cell).covers owner address) : a.output.backing.covers owner address := by
  rcases covered with ⟨oe,lo,hi⟩
  simp only [key,Allocation.key] at oe lo hi
  have inside:=a.output.active_byte (offset (shape t s) cell)
    (address-(a.output.window.baseByte+4*offset (shape t s) cell))
    (output_inside ok cell (touched_complete ok cell live)) (by omega)
  convert inside.1 using 1 <;> omega

def readBackings (a : Binding t) : List ByteAllocation :=
  [a.input.backing,a.weight.backing,a.starts.backing,a.lengths.backing,a.indices.backing]

theorem caller_frame (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (lists a before))
    (backing : ByteAllocation) (read : backing∈readBackings a) (owner address : Nat)
    (covered : backing.covers owner address) : sourceStores rest ops t s a before owner address=before owner address := by
  have c:=(python_checks_iff t s (lists a before)).mp ok
  have hx:=a.inputSeparate (c.separate t.x (by simp [reads])).symm
  have hw:=a.weightSeparate (c.separate t.w (by simp [reads])).symm
  have hs:=a.startsSeparate (c.separate t.starts (by simp [reads])).symm
  have hl:=a.lengthsSeparate (c.separate t.lengths (by simp [reads])).symm
  have hi:=a.indicesSeparate (c.separate t.indices (by simp [reads])).symm
  have separated : backing.separate a.output.backing := by
    simp only [readBackings,List.mem_cons,List.not_mem_nil,or_false] at read
    rcases read with rfl|rfl|rfl|rfl|rfl <;> assumption
  apply source_frame
  intro cell live cellCovered
  exact VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.separate_byte _ _ separated owner address covered
    (output_span ok cell live owner address cellCovered)

/-- Any partially issued subset/order has the exact complement frame, even
when failure prevents a numerical statement about issued store payloads. -/
theorem failure_frame (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (issued : List (Nat×Nat)) (payload : Nat×Nat→Raw.Word)
    (actual : ∀ cell∈issued,Touched (shape t s) (segments t s (lists a before)) cell)
    (owner address : Nat) (outside : ∀ cell,Touched (shape t s) (segments t s (lists a before)) cell →
      ¬(key a s cell).covers owner address) :
    stores Raw.ops (key a s) payload issued before owner address=before owner address := by
  apply bytes_frame Raw.ops
  intro cell member
  exact outside cell (actual cell member)

theorem empty (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (empty : (shape t s).columns=0 ∨ ∀ segment∈segments t s (lists a before),¬segment.active) :
    sourceStores rest ops t s a before=before := by
  have none : cells (shape t s) (segments t s (lists a before))=[] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro cell member
    rcases ((mem_cells _ _ _).mp member).2.2 with ⟨segment,hs,active,_,_,lo,hi⟩
    rcases empty with zero|inactive
    · omega
    · exact inactive segment hs active
  simp [sourceStores,none,stores]

theorem addresses (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (lists a before)) (segment : Segment)
    (member : segment∈segments t s (lists a before)) (active : segment.active)
    (hn : 0<(shape t s).columns) (hk : 0<(shape t s).inner)
    (slot : Nat) (selected : segment.adapter=some slot) (r c k : Nat) :
    (BitVec.ofNat 64 a.input.window.baseByte+BitVec.ofNat 64 (Bounds.inputOffset (shape t s) segment r k)*BitVec.ofNat 64 4).toNat=
      t.x.dataPtr.toNat+4*Bounds.inputOffset (shape t s) segment r k ∧
    (BitVec.ofNat 64 a.weight.window.baseByte+BitVec.ofNat 64 (Bounds.weightOffset (shape t s) slot c k)*BitVec.ofNat 64 2).toNat=
      t.w.dataPtr.toNat+2*Bounds.weightOffset (shape t s) slot c k ∧
    (BitVec.ofNat 64 a.output.window.baseByte+BitVec.ofNat 64 (Bounds.outputOffset (shape t s) segment r c)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*Bounds.outputOffset (shape t s) segment r c := by
  have h:=Bounds.formed ok segment member active hn hk slot selected r c k
  exact ⟨a.input.pointer_word _ h.1 (by decide),a.weight.pointer_word _ h.2.1 (by decide),a.output.pointer_word _ h.2.2 (by decide)⟩

theorem output_address (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (lists a before)) (segment : Segment)
    (member : segment∈segments t s (lists a before)) (active : segment.active)
    (hn : 0<(shape t s).columns) (r c : Nat) :
    (BitVec.ofNat 64 a.output.window.baseByte+BitVec.ofNat 64 (Bounds.outputOffset (shape t s) segment r c)*BitVec.ofNat 64 4).toNat=
      t.y.dataPtr.toNat+4*Bounds.outputOffset (shape t s) segment r c :=
  a.output.pointer_word _ (Bounds.output_formed ok segment member active hn r c) (by decide)

theorem zero_batches (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (lists a before)) (zero : s.batches=0) :
    sourceStores rest ops t s a before=before := by
  have c:=(python_checks_iff t s (lists a before)).mp ok
  have count:=(lists a before).startCount.trans c.shapes.2.2.2.1
  have starts : (lists a before).starts=[] := by simpa [zero] using count
  have noSegments : segments t s (lists a before)=[] := by simp [segments,raw,HostMetadata.rawSegments,starts]
  apply empty rest ops t s a before
  right
  intro segment member
  simp [noSegments] at member

theorem metadata_spans (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (lists a before)) (b : Nat) (hb : b<s.batches.toNat) :
    (∀ byte : Fin (indexBytes t.starts),a.starts.backing.covers a.starts.window.owner
      (a.starts.window.baseByte+indexBytes t.starts*b+byte.val)) ∧
    (∀ byte : Fin (indexBytes t.lengths),a.lengths.backing.covers a.lengths.window.owner
      (a.lengths.window.baseByte+indexBytes t.lengths*b+byte.val)) ∧
    (∀ byte : Fin (indexBytes t.indices),a.indices.backing.covers a.indices.window.owner
      (a.indices.window.baseByte+indexBytes t.indices*b+byte.val)) := by
  have c:=(python_checks_iff t s (lists a before)).mp ok
  have hs : b<t.starts.numel := by rw [c.shapes.2.2.2.1];exact hb
  have hl : b<t.lengths.numel := hb.trans_le c.shapes.2.2.2.2.2
  have hi : b<t.indices.numel := by rw [c.shapes.2.2.2.2.1];exact hb
  exact ⟨fun byte => (a.starts.active_byte b byte.val hs byte.isLt).1,
    fun byte => (a.lengths.active_byte b byte.val hl byte.isLt).1,
    fun byte => (a.indices.active_byte b byte.val hi byte.isLt).1⟩

structure Result (y : TensorInfo) where
  tensor : TensorInfo
  same : tensor=y
  returned : Unit
  bytes : Raw.State

def invoke (rest : Algebra Raw.Word) (ops : Native.Primitives)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (_ok : Admitted t s (lists a before)) : Result t.y :=
  ⟨t.y,rfl,(),sourceStores rest ops t s a before⟩

#axiomsClean output_span
#axiomsClean caller_frame
#axiomsClean failure_frame
#axiomsClean empty
#axiomsClean addresses
#axiomsClean metadata_spans
#axiomsClean output_address
#axiomsClean zero_batches
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Direct.NativeCall
