import bench.tritonbench_g.fused_layernorm_triton.Proofs.Source
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Binding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
namespace Torch
export VeriTile.Torch.FP32 (Tensor TensorLaws Storage)
end Torch
noncomputable section

/-- Stable backing identity. Aliased inputs use exactly the same region. -/
def region (t : Torch.Tensor) : RegionName := "torch-storage-" ++ toString t.storage.object

/-- Coordinate descriptor for a contiguous Tensor. Singleton physical strides
need not be canonical: TensorLaws.contiguous proves its actual coordinates.
The original Tensor object token and its backing identity are retained. -/
def descriptor (t : Torch.Tensor) (strides : List Nat) : Whole.Tensor :=
  ⟨t.object, region t, t.metadata.shape, strides, .fp32, t.storageOffset, t.metadata.device.index⟩

structure MetadataBinding (m : Contract.Metadata) (t : Torch.Tensor) : Prop where
  metadata : t.metadata = m.toTensorMetadata
  offset : t.storageOffset = m.storageOffset

/-- Ordinary typed-memory interpretation. This only binds input storage cells;
it has no assertion about kernel execution or any normalization result. -/
structure MemoryBinding {α : Type} (s : State α) (read : Nat → α) (t : Torch.Tensor) : Prop where
  cells : ∀ i, i < t.storage.elements → s.mem (region t) i = .mk .real (read (t.storage.address i))

def logicalRead {α : Type} (read : Nat → α) (t : Torch.Tensor) (i : Nat) : α :=
  read (t.storage.address (t.physicalIndex i))

theorem coordinate {α : Type} (s : State α) (read : Nat → α) (m : Contract.Metadata)
    (t : Torch.Tensor) (api : Torch.TensorLaws t) (bound : MetadataBinding m t)
    (memory : MemoryBinding s read t) (h : Contract.TensorContract m) (i : Nat) (hi : i < m.numel) :
    s.mem (region t) i = .mk .real (logicalRead read t i) := by
  have hc : t.metadata.isContiguous = Bool.true := by rw [bound.metadata]; exact h.contiguous
  have hn : i < t.metadata.numel := by simpa [bound.metadata] using hi
  have off : t.storageOffset = 0 := bound.offset.trans h.offset
  have physical := api.contiguous hc i hn
  rw [off, Nat.zero_add] at physical
  have inBounds := api.inBounds i hn
  rw [physical] at inBounds
  simpa [logicalRead, physical] using memory.cells i inBounds

theorem typed {α : Type} (s : State α) (read : Nat → α) (m : Contract.Metadata)
    (t : Torch.Tensor) (api : Torch.TensorLaws t) (bound : MetadataBinding m t)
    (memory : MemoryBinding s read t) (h : Contract.TensorContract m) (i : Nat) (hi : i < m.numel) :
    ∃ v, s.mem (region t) i = .mk .real v :=
  ⟨logicalRead read t i, coordinate s read m t api bound memory h i hi⟩

theorem whole_domain {α : Type} (s : State α) (read : Nat → α) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (h : Contract.InputContract w b x)
    (wx : MetadataBinding w W) (bx : MetadataBinding b B) (xx : MetadataBinding x X)
    (wa : Torch.TensorLaws W) (ba : Torch.TensorLaws B) (xa : Torch.TensorLaws X)
    (wm : MemoryBinding s read W) (bm : MemoryBinding s read B) (xm : MemoryBinding s read X) :
    Whole.InputContract s (descriptor X [Contract.width x,1]) (descriptor W [1]) (descriptor B [1])
      (Contract.rows x) (Contract.width x) := by
  have xs := Contract.input_shape x h.rank
  refine ⟨h.positive, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [descriptor, xx.metadata, xx.offset, xs, h.input.offset, h.input.device]
  · simp [descriptor, wx.metadata, wx.offset, h.weightShape, h.weight.offset, h.weight.device]
  · simp [descriptor, bx.metadata, bx.offset, h.biasShape, h.bias.offset, h.bias.device]
  · intro row hr col hc
    apply typed s read x X xa xx xm h.input
    simpa [VeriTile.Torch.TensorMetadata.numel, xs] using output_bounds _ _ _ _ hr hc
  · intro col hc
    apply typed s read w W wa wx wm h.weight
    simpa [VeriTile.Torch.TensorMetadata.numel, h.weightShape] using hc
  · intro col hc
    apply typed s read b B ba bx bm h.bias
    simpa [VeriTile.Torch.TensorMetadata.numel, h.biasShape] using hc

/-- Counts fit because FP32 data occupies an actual 64-bit address span. This
is an ordinary Tensor/storage law, not an extra small-shape admission guard. -/
theorem count_bound (m : Contract.Metadata) (t : Torch.Tensor) (api : Torch.TensorLaws t)
    (bound : MetadataBinding m t) (h : Contract.TensorContract m) : 4 * m.numel ≤ 2^64 := by
  have hc : t.metadata.isContiguous = Bool.true := by rw [bound.metadata]; exact h.contiguous
  have span := (VeriTile.Torch.FP32.dataView t api hc).spanFits
  change _ + 4 * t.metadata.numel ≤ 2^64 at span
  rw [bound.metadata] at span
  omega

theorem counts (w b x : Contract.Metadata) (W X : Torch.Tensor) (h : Contract.InputContract w b x)
    (wx : MetadataBinding w W) (xx : MetadataBinding x X)
    (wa : Torch.TensorLaws W) (xa : Torch.TensorLaws X) :
    Contract.width x ≤ 2^62 ∧ Contract.rows x * Contract.width x ≤ 2^62 := by
  have wb := count_bound w W wa wx h.weight
  have xb := count_bound x X xa xx h.input
  simp only [VeriTile.Torch.TensorMetadata.numel, h.weightShape, Contract.input_shape x h.rank,
    List.prod_cons, List.prod_nil, Nat.mul_one] at wb xb
  constructor <;> omega

theorem actual_pointer (m : Contract.Metadata) (t : Torch.Tensor) (api : Torch.TensorLaws t)
    (bound : MetadataBinding m t) (h : Contract.TensorContract m) (nonempty : 0 < m.numel) :
    m.dataPtr = BitVec.ofNat 64 t.storage.base := by
  have hn : 0 < t.metadata.numel := by simpa [bound.metadata] using nonempty
  simpa [VeriTile.Torch.FP32.Tensor.dataPtr, bound.metadata, bound.offset, h.offset,
    VeriTile.Torch.FP32.Storage.address] using api.pointer hn

/-- Only ordinary Tensor, typed-memory and fresh allocator facts supplement
admission. In particular this does not assume a successful kernel or output. -/
structure Inputs {α : Type} (s : State α) (read : Nat → α) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) : Prop where
  weightMetadata : MetadataBinding w W
  biasMetadata : MetadataBinding b B
  inputMetadata : MetadataBinding x X
  weightAPI : Torch.TensorLaws W
  biasAPI : Torch.TensorLaws B
  inputAPI : Torch.TensorLaws X
  weightMemory : MemoryBinding s read W
  biasMemory : MemoryBinding s read B
  inputMemory : MemoryBinding s read X

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Binding
