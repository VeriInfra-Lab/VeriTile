import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Host
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves Materialized Allocation FloatArray Produced TensorArithmetic RowLayout value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor SameMetadata)
noncomputable section

/-- Contiguous views can keep a nonzero storage offset. This is weaker than a
fresh dense allocation and covers ordinary Tensor.contiguous() identity returns. -/
def Packed (t : Tensor) : Prop :=
  t.isNeg=Bool.false ∧ ∀ i, i < t.numel → t.index i=t.storageOffset+i

structure Arithmetic (α : Type) extends TensorArithmetic α where
  multiply : α → α → α

/-- Only ordinary, coordinate-level tensor/runtime contracts. The int32
extensions add the selected label dtype without changing the frozen int64
helper. Every reduction takes its explicit arithmetic interpreter. -/
structure Library {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : Arithmetic α) (rt : Runtime α σ) : Prop where
  base : VeriTile.Bench.Repairs.CrossEntropy1.Host.Library obs rt.toRuntime
  operations : VeriTile.Bench.Repairs.CrossEntropy1.Host.TensorLibrary obs A.toTensorArithmetic rt.toRuntime
  resolveInt32 : ∀ input before output after, before.live input.region → input.dtype="torch.int32" →
    rt.resolveNeg input before=.ok output after → ∀ i, i < input.numel → obs.integer after output i=obs.integer before input i
  contiguousInt32 : ∀ input before output after, before.live input.region → input.dtype="torch.int32" →
    rt.contiguous input before=.ok output after → ∀ i, i < input.numel → obs.integer after output i=obs.integer before input i
  contiguousPacked : ∀ input before output after, before.live input.region → input.isNeg=Bool.false →
    rt.contiguous input before=.ok output after → Packed output
  int32Raw : ∀ input state i, state.live input.region → input.dtype="torch.int32" →
    Packed input → i < input.numel → (state.memory input.region (input.storageOffset+i)).read .int=obs.integer state input i
  setupForward : ∀ p count before after, rt.setupForward2 p count before=.ok () after → Preserves obs before after
  setupBackward : ∀ p count before after, rt.setupBackward2 p count before=.ok () after → Preserves obs before after
  multiply : ∀ left right before output after,
    before.live left.region → before.live right.region → left.dtype="torch.float32" → right.dtype="torch.float32" →
    left.shape=right.shape → rt.multiply left right before=.ok output after →
      Produced obs before after output left.shape left.device
        (fun i => A.multiply (value obs before left i) (value obs before right i))
  castInt64 : ∀ input before output after, before.live input.region → input.dtype="torch.int32" →
    rt.toInt64 input before=.ok output after →
      Allocation obs output input.shape "torch.int64" input.device before after ∧ output.Dense ∧
      ∀ i, i < input.numel → obs.integer after output i=obs.integer before input i

structure Normalized {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (input output : Tensor) (before after : World α σ) : Prop where
  materialized : Materialized obs input output before after
  packed : Packed output
  rows : RowLayout output
  int32 : input.dtype="torch.int32" → ∀ i, i < input.numel → obs.integer after output i=obs.integer before input i

/-- Successful normalization is extracted from the actual resolve/contiguous
calls; no pre-normalized-input or successful-host-output premise is used. -/
theorem normalize_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (input output : Tensor) (before after : World α σ)
    (live : before.live input.region) (success : normalize rt input before=.ok output after) :
    Normalized obs input output before after := by
  obtain ⟨resolved,intermediate,resolveCall,contiguousCall⟩ :=
    (bind_success _ _ _ _).mp success
  have first := lib.base.resolve input before resolved intermediate live resolveCall
  have second := lib.base.contiguous resolved intermediate output after first.1.live contiguousCall
  have metadata : SameMetadata input output :=
    ⟨second.1.metadata.shape.trans first.1.metadata.shape,
     second.1.metadata.dtype.trans first.1.metadata.dtype,
     second.1.metadata.cuda.trans first.1.metadata.cuda,
     second.1.metadata.device.trans first.1.metadata.device⟩
  refine ⟨⟨metadata,first.1.frame.trans second.1.frame,second.1.live,?_,?_,second.1.rank⟩,
    lib.contiguousPacked _ _ _ _ first.1.live first.2 contiguousCall,second.2.1,?_⟩
  · intro dtype i bound
    exact (second.1.floats (by simpa only [first.1.metadata.dtype] using dtype) i
      (by simpa only [first.1.metadata.numel] using bound)).trans (first.1.floats dtype i bound)
  · intro dtype i bound
    exact (second.1.integers (by simpa only [first.1.metadata.dtype] using dtype) i
      (by simpa only [first.1.metadata.numel] using bound)).trans (first.1.integers dtype i bound)
  · intro dtype i bound
    exact (lib.contiguousInt32 _ _ _ _ first.1.live
      (by simpa only [first.1.metadata.dtype] using dtype) contiguousCall i
      (by simpa only [first.1.metadata.numel] using bound)).trans
        (lib.resolveInt32 _ _ _ _ live dtype resolveCall i bound)

#axiomsClean normalize_values
#specNonCircular Library avoiding [Source.forward,Source.backward,Host.forward,Host.backward,Host.backwardGrid,Grid.forward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
