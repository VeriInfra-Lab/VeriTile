import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Geometry
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.NativeInstructions

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Lane (r : Request) := GridIndex r.call.grid × TileIndex [r.call.layout.BM,r.call.layout.D] × Fin 4

def active (r : Request) (i : Lane r) : Prop := FPReference.laneActive r.call i.1 i.2.1
def address (r : Request) (i : Lane r) : Nat :=
  r.out.tensor.dataPtr.toNat+4*FPReference.laneOffset r.call i.1 i.2.1+i.2.2.val

theorem address_unique (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (i j : Lane r) (hi : active r i) (hj : active r j) (he : address r i=address r j) : i=j := by
  have il := i.2.2.isLt
  have jl := j.2.2.isLt
  have offsets : FPReference.laneOffset r.call i.1 i.2.1=FPReference.laneOffset r.call j.1 j.2.1 := by
    unfold address at he; omega
  let hm := layout_memory r heap h
  have ia : RealIO.active r.call.layout (r.call.launch i.1) (Program.inputs (metadataState r heap) regions) i.2.1 :=
    (Host.active_iff hm i.1 i.2.1).mpr hi
  have ja : RealIO.active r.call.layout (r.call.launch j.1) (Program.inputs (metadataState r heap) regions) j.2.1 :=
    (Host.active_iff hm j.1 j.2.1).mpr hj
  have equal : RealIO.write r.call.layout (r.call.launch i.1) (Program.inputs (metadataState r heap) regions) i.2.1=
      RealIO.write r.call.layout (r.call.launch j.1) (Program.inputs (metadataState r heap) regions) j.2.1 := by
    rw [Host.write_eq hm,Host.write_eq hm]
    exact offsets
  obtain ⟨sameProgram,sameLane⟩ := Host.unique_writer hm i.1 j.1 i.2.1 j.2.1 ia ja equal
  have sameByte : i.2.2=j.2.2 := Fin.ext (by unfold address at he; omega)
  exact Prod.ext sameProgram (Prod.ext sameLane sameByte)

def writeback {α : Type} (r : Request) (values : Lane r → α) (before : Nat → α) (a : Nat) : α :=
  if h : ∃ i : Lane r,active r i ∧ address r i=a then values (Classical.choose h) else before a

theorem writeback_readback {α : Type} (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (values : Lane r → α) (before : Nat → α) (i : Lane r) (hi : active r i) :
    writeback r values before (address r i)=values i := by
  have ex : ∃ j : Lane r,active r j ∧ address r j=address r i := ⟨i,hi,rfl⟩
  rw [writeback,dif_pos ex]
  have selected := Classical.choose_spec ex
  rw [address_unique r heap h (Classical.choose ex) i selected.1 hi selected.2]

theorem writeback_frame {α : Type} (r : Request) (values : Lane r → α) (before : Nat → α) (a : Nat)
    (outside : ¬∃ i : Lane r,active r i ∧ address r i=a) : writeback r values before a=before a := by
  rw [writeback,dif_neg outside]

theorem active_span (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (i : Lane r) (hi : active r i) :
    r.out.tensor.storage.base≤address r i ∧
      address r i<r.out.tensor.storage.base+r.out.storageBytes := by
  have flat := flat_index r.call regions (extent r) h.checks.2.1 i.1 i.2.1 hi
  have descriptor := h.checks.1.1 (views r).o (by simp)
  have shape : r.out.tensor.metadata.shape=[r.call.q.rows,r.call.q.heads,r.call.q.dims] := descriptor.1
  have contiguous : r.out.tensor.metadata.isContiguous=Bool.true := descriptor.2.2.1
  have bound : flatIndex r.call i.1 i.2.1<r.out.tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using flat.2
  unfold address
  rw [flat.1]
  exact payload_span r.out h.outAPI contiguous _ bound i.2.2

/-- The full untyped read backing, including bytes preceding/following views. -/
def ReadFrame {α : Type} (r : Request) (before after : Nat → α) : Prop :=
  ∀ x∈Checked.reads (views r),∀ off,off < x.storageBytes →
    after (x.storageBase+off)=before (x.storageBase+off)

theorem writeback_inputs {α : Type} (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (values : Lane r → α) (before : Nat → α) : ReadFrame r before (writeback r values before) := by
  intro x hx off hoff
  apply writeback_frame
  rintro ⟨i,hi,he⟩
  have span := active_span r heap h i hi
  have hb := Host.launch_batch_lt r.call i.1
  have ne : r.call.batches≠0 := by omega
  have separated := (h.checks.1.2.2.2.2.resolve_left ne) x hx
  change Checked.separate r.out.view x at separated
  unfold Checked.separate Payload.view at separated
  simp only at separated
  rcases separated with emptyO|emptyR|beforeO|beforeR <;> omega

def Needed (r : Request) : Prop := ∃ b,b < r.call.batches ∧ 0 < r.call.length b
def order (r : Request) : List (GridIndex r.call.grid) :=
  if Needed r then FPHost.jobs r.call else []

theorem empty_writeback {α : Type} (r : Request) (values : Lane r → α) (before : Nat → α)
    (empty : ¬Needed r) : writeback r values before=before := by
  funext a
  apply writeback_frame
  rintro ⟨i,hi,_⟩
  exact empty ⟨(r.call.launch i.1).batch,Host.launch_batch_lt r.call i.1,Nat.zero_lt_of_lt hi⟩

#specNonCircular writeback avoiding [FP.Structural.exec,FPReference.output,FPReference.grid]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
