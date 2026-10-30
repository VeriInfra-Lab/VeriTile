import bench.tritonbench_g.cross_entropy1.Proofs.HostTrace

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor SameMetadata)
noncomputable section

/-- Logical integer reads belong to the tensor runtime, so an int64 caller
view can share bytes with FP32 logits. Only fresh cloned labels enter the
kernel's tagged int64 channel. This avoids a false no-alias precondition. -/
structure Observations (α σ : Type) where
  neg : α → α
  integer : World α σ → Tensor → Nat → Int

def value {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (s : World α σ) (t : Tensor) (i : Nat) : α :=
  let raw := (s.memory t.region (t.index i)).read .real
  if t.isNeg then obs.neg raw else raw

structure Preserves {α σ : Type} (obs : Observations α σ) (before after : World α σ) : Prop where
  live : ∀ r, before.live r → after.live r
  memory : ∀ r i, before.live r → after.memory r i=before.memory r i
  integers : ∀ t i, before.live t.region → obs.integer after t i=obs.integer before t i

theorem Preserves.refl {α σ : Type} (obs : Observations α σ) (s : World α σ) : Preserves obs s s :=
  ⟨fun _ h => h,fun _ _ _ => rfl,fun _ _ _ => rfl⟩

theorem Preserves.trans {α σ : Type} {obs : Observations α σ} {a b c : World α σ}
    (ab : Preserves obs a b) (bc : Preserves obs b c) : Preserves obs a c :=
  ⟨fun r h => bc.live r (ab.live r h),
   fun r i h => (bc.memory r i (ab.live r h)).trans (ab.memory r i h),
   fun t i h => (bc.integers t i (ab.live t.region h)).trans (ab.integers t i h)⟩

theorem Preserves.floatValue {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {a b : World α σ} (preserved : Preserves obs a b) (t : Tensor) (live : a.live t.region) (i : Nat) :
    value obs b t i=value obs a t i := by
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Host.value
  rw [preserved.memory t.region (t.index i) live]

structure Materialized {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (input output : Tensor) (before after : World α σ) : Prop where
  metadata : SameMetadata input output
  frame : Preserves obs before after
  live : after.live output.region
  floats : input.dtype="torch.float32" → ∀ i, i < input.numel → value obs after output i=value obs before input i
  integers : input.dtype="torch.int64" → ∀ i, i < input.numel →
    obs.integer after output i=obs.integer before input i
  rank : output.strides.length=output.shape.length

structure Allocation {α σ : Type} (obs : Observations α σ) (output : Tensor)
    (shape : List Nat) (dtype : String) (device : Nat) (before after : World α σ) : Prop where
  shapeEq : output.shape=shape
  dtypeEq : output.dtype=dtype
  deviceEq : output.device=device
  cuda : output.isCUDA=Bool.true
  notNeg : output.isNeg=Bool.false
  frame : Preserves obs before after
  live : after.live output.region
  fresh : ¬before.live output.region
  rank : output.strides.length=output.shape.length

/-- Pure logical row-layout condition, including singleton axes whose stored
stride may differ from one without changing a single addressed coordinate. -/
def RowLayout (t : Tensor) : Prop := ∀ rows cols, t.shape=[rows,cols] →
  ∀ row col, row<rows → col<cols → t.index (row*cols+col)=t.storageOffset+row*rowStride t+col

/-- Ordinary local tensor contracts. No field mentions a cross-entropy
formula, a whole call's result, a source-grid output, or execution success. -/
structure Library {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ) : Prop where
  resolve : ∀ input before output after, before.live input.region →
    rt.resolveNeg input before=.ok output after → Materialized obs input output before after ∧ output.isNeg=Bool.false
  clone : ∀ input before output after, before.live input.region → input.isNeg=Bool.false →
    rt.cloneContiguous input before=.ok output after →
    Materialized obs input output before after ∧ output.Dense ∧ ¬before.live output.region ∧
    (input.dtype="torch.int64" → ∀ i, i < input.numel →
      (after.memory output.region i).read .int=obs.integer before input i)
  contiguous : ∀ input before output after, before.live input.region →
    rt.contiguous input before=.ok output after → Materialized obs input output before after ∧ RowLayout output ∧
      (input.isNeg=Bool.false → output.isNeg=Bool.false)
  empty : ∀ shape dtype device before output after,
    rt.empty shape dtype device before=.ok output after →
      Allocation obs output shape dtype device before after ∧ output.Dense
  emptyLike : ∀ input packed before output after, before.live input.region →
    rt.emptyLike input packed before=.ok output after →
      Allocation obs output input.shape input.dtype input.device before after ∧
      (packed=Bool.true → output.Dense) ∧
      (∀ rows cols, output.shape=[rows,cols] → lastStride output=1 → rows≤1 ∨ cols≤rowStride output)
  enter : ∀ device before after, rt.enterDevice device before=.ok () after → Preserves obs before after
  exit : ∀ device before after, rt.exitDevice device before=.ok () after → Preserves obs before after
  setupForward : ∀ p n before after, rt.setupForward p n before=.ok () after → Preserves obs before after
  setupBackward : ∀ p n before after, rt.setupBackward p n before=.ok () after → Preserves obs before after
  save : ∀ x l y original before after, rt.saveTensors x l y original before=.ok () after → Preserves obs before after
  copy : ∀ dst src before out after, before.live dst.region → before.live src.region → dst.shape=src.shape →
    rt.copy dst src before=.ok out after →
      out=dst ∧ (∀ r, before.live r → after.live r) ∧
      (∀ i, i<dst.numel → value obs after dst i=value obs before src i) ∧
      (∀ r i, (r≠dst.region ∨ ∀ j, j<dst.numel → i≠dst.index j) → after.memory r i=before.memory r i)

structure SnapshotValues {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (input output : Tensor) (before after : World α σ) : Prop where
  materialized : Materialized obs input output before after
  dense : output.Dense
  fresh : ¬before.live output.region
  labels : input.dtype="torch.int64" → ∀ i, i < input.numel →
    (after.memory output.region i).read .int=obs.integer before input i

theorem snapshot_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (input output : Tensor) (before after : World α σ)
    (live : before.live input.region) (success : snapshot rt input before=.ok output after) :
    SnapshotValues obs input output before after := by
  obtain ⟨trace⟩ := snapshot_success rt input output before after success
  have resolved := lib.resolve _ _ _ _ live trace.resolveCall
  have cloned := lib.clone _ _ _ _ resolved.1.live resolved.2 trace.cloneCall
  have metadata : SameMetadata input output :=
    ⟨cloned.1.metadata.shape.trans resolved.1.metadata.shape,
     cloned.1.metadata.dtype.trans resolved.1.metadata.dtype,
     cloned.1.metadata.cuda.trans resolved.1.metadata.cuda,
     cloned.1.metadata.device.trans resolved.1.metadata.device⟩
  have floatValues : input.dtype="torch.float32" → ∀ i, i < input.numel →
      value obs after output i=value obs before input i := by
    intro dtype i bound
    exact (cloned.1.floats (by simpa [resolved.1.metadata.dtype] using dtype) i
      (by simpa [resolved.1.metadata.numel] using bound)).trans (resolved.1.floats dtype i bound)
  have intValues : input.dtype="torch.int64" → ∀ i, i < input.numel →
      obs.integer after output i=obs.integer before input i := by
    intro dtype i bound
    exact (cloned.1.integers (by simpa [resolved.1.metadata.dtype] using dtype) i
      (by simpa [resolved.1.metadata.numel] using bound)).trans (resolved.1.integers dtype i bound)
  refine ⟨⟨metadata,resolved.1.frame.trans cloned.1.frame,cloned.1.live,floatValues,intValues,cloned.1.rank⟩,
    cloned.2.1,?_,?_⟩
  · intro old
    exact cloned.2.2.1 (resolved.1.frame.live _ old)
  · intro dtype i bound
    rw [cloned.2.2.2 (by simpa [resolved.1.metadata.dtype] using dtype) i
      (by simpa [resolved.1.metadata.numel] using bound),resolved.1.integers dtype i bound]

#axiomsClean snapshot_values
#axiomsClean Preserves.trans
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
