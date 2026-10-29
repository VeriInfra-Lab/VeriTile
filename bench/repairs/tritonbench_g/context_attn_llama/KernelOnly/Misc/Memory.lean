import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Geometry
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeInstructions
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPReferenceGrid

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
open Classical
set_option maxHeartbeats 4000000

abbrev Lane (r : Request) := GridIndex r.call.grid × TileIndex [r.call.layout.BM,r.call.layout.D] × Fin 4

def active (r : Request) (i : Lane r) : Prop := r.call.active i.1 i.2.1
def address (r : Request) (i : Lane r) : Nat :=
  r.out.tensor.dataPtr.toNat+4*r.call.offset i.1 i.2.1+i.2.2.val

theorem address_unique (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (i j : Lane r) (hi : active r i) (hj : active r j) (he : address r i=address r j) : i=j := by
  have il := i.2.2.isLt
  have jl := j.2.2.isLt
  have offsets : r.call.offset i.1 i.2.1=r.call.offset j.1 j.2.1 := by
    unfold address at he; omega
  obtain ⟨sameProgram,sameLane⟩ := Host.writer_unique r.call (allocation r) (metadataState r heap)
    (layout_valid r heap h) i.1 j.1 i.2.1 j.2.1 hi hj offsets
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
  have flat := flat_index r.call r.dimensions (extent r) h.checks.2.1 i.1 i.2.1 hi
  have descriptor := h.checks.1.1 (views r).o (by simp)
  have shape : r.out.tensor.metadata.shape=[r.dimensions.queryRows,r.call.layout.heads,r.call.layout.D] := descriptor.1
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
  have hb := Host.batch_bound r.call i.1
  have ne : r.call.batches≠0 := by omega
  have separated := (h.checks.1.2.2.2.2.2.2.resolve_left ne) x hx
  change Checked.separate r.out.view x at separated
  unfold Checked.separate Payload.view at separated
  simp only at separated
  rcases separated with emptyO|emptyR|beforeO|beforeR <;> omega

def Needed (r : Request) : Prop := ∃ b,b < r.call.batches ∧ 0 < r.call.length b
def order (r : Request) : List (GridIndex r.call.grid) :=
  if Needed r then FPSuccess.allPrograms r.call else []

theorem empty_writeback {α : Type} (r : Request) (values : Lane r → α) (before : Nat → α)
    (empty : ¬Needed r) : writeback r values before=before := by
  funext a
  apply writeback_frame
  rintro ⟨i,hi,_⟩
  exact empty ⟨(Formula.batch r.call.layout (r.call.launchAt i.1)),Host.batch_bound r.call i.1,Nat.zero_lt_of_lt hi⟩

#specNonCircular writeback avoiding [FP.Structural.exec,FPReference.output,FPReference.grid]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
