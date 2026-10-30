import bench.tritonbench_g.cross_entropy_ops.Proofs.KernelContract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelBinding
open VeriTile.Triton FP.Structural KernelContract
open CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Ordinary API adapter from the existing strided descriptor to the shared
FP32 Tensor object. The index law applies only to valid logical elements;
empty tensors and unused backing bytes gain no alignment or capacity premise. -/
structure Binding (alloc : RegionName → Allocation) (descriptor : Tensor)
    (actual : VeriTile.Torch.FP32.Tensor) : Prop where
  laws : VeriTile.Torch.FP32.TensorLaws actual
  object : actual.object=descriptor.objectId
  shape : actual.metadata.shape=descriptor.shape
  offset : actual.storageOffset=descriptor.storageOffset
  physical : ∀ i, i<descriptor.numel → actual.physicalIndex i=descriptor.index i
  base : actual.storage.base=(alloc descriptor.region).base
  capacity : actual.storage.elements=(alloc descriptor.region).capacity/4
  cuda : actual.metadata.device.kind="cuda"
  device : actual.metadata.device.index=descriptor.device
  noNegative : actual.metadata.isNeg=Bool.false
  pointer : 0<descriptor.numel → actual.dataPtr.toNat=(alloc descriptor.region).base+4*descriptor.storageOffset

def read {α : Type} [Inhabited α] (initial : State α) (region : RegionName)
    (actual : VeriTile.Torch.FP32.Tensor) (i : Nat) : α :=
  (initial.mem region (actual.physicalIndex i)).read .real

theorem read_eq {α : Type} [Inhabited α] (alloc) (descriptor : Tensor)
    (actual : VeriTile.Torch.FP32.Tensor) (binding : Binding alloc descriptor actual)
    (initial : State α) (i : Nat) (hi : i<descriptor.numel) :
    read initial descriptor.region actual i=fpValue initial descriptor i := by
  simp only [read,binding.physical i hi,fpValue]

theorem physical_address (alloc) (descriptor : Tensor) (actual)
    (binding : Binding alloc descriptor actual) (i : Nat) (hi : i<descriptor.numel) :
    actual.storage.address (actual.physicalIndex i)=(alloc descriptor.region).base+4*descriptor.index i := by
  simp only [VeriTile.Torch.FP32.Storage.address,binding.base,binding.physical i hi]

theorem initial_word (alloc bytes) (descriptor : Tensor) (actual)
    (binding : Binding alloc descriptor actual) (words : State Word) (pointer : Nat)
    (native : FloatBinding alloc bytes words descriptor pointer) (i : Nat) (hi : i<descriptor.numel) :
    read words descriptor.region actual i=wordAt (bytes descriptor.region) (4*actual.physicalIndex i) := by
  rw [read_eq alloc descriptor actual binding words i hi,binding.physical i hi]
  exact native.cells i hi

theorem scalar_numel (t : Tensor) (region : RegionName) (base rows blocks row block : Nat)
    (view : ScalarOutputView t region base rows blocks) (hr : row<rows) (hb : block<blocks) :
    block*rows+row<t.numel := by
  by_cases many : 1<blocks
  · have bound := Nat.mul_le_mul_right rows (Nat.succ_le_of_lt hb)
    simp only [Nat.succ_mul] at bound
    simp only [Tensor.numel,view.shape,if_pos many,List.prod_cons,List.prod_nil,Nat.mul_one]
    omega
  · have zero : block=0 := by omega
    simpa [Tensor.numel,view.shape,many,zero] using hr

theorem scalar_read {α : Type} [Inhabited α] (alloc) (t : Tensor) (actual)
    (binding : Binding alloc t actual) (initial : State α)
    (region : RegionName) (base rows blocks row block : Nat)
    (view : ScalarOutputView t region base rows blocks) (hr : row<rows) (hb : block<blocks) :
    read initial region actual (block*rows+row)=fpValue initial t (block*rows+row) := by
  simpa only [view.regionEq] using read_eq alloc t actual binding initial _ (scalar_numel _ _ _ _ _ _ _ view hr hb)

theorem matrix_read {α : Type} [Inhabited α] (alloc) (t : Tensor) (actual)
    (binding : Binding alloc t actual) (initial : State α)
    (region : RegionName) (base rows cols stride row col : Nat)
    (view : MatrixView t region base rows cols stride) (hr : row<rows) (hc : col<cols) :
    read initial region actual (row*cols+col)=fpValue initial t (row*cols+col) := by
  simpa only [view.regionEq] using read_eq alloc t actual binding initial _ (matrix_numel _ _ _ _ _ _ _ _ view hr hc)

theorem vector_read {α : Type} [Inhabited α] (alloc) (t : Tensor) (actual)
    (binding : Binding alloc t actual) (initial : State α)
    (region : RegionName) (base count stride i : Nat)
    (view : VectorView t region base count stride) (hi : i<count) :
    read initial region actual i=fpValue initial t i := by
  have bound : i<t.numel := by simpa [Tensor.numel,view.shape] using hi
  simpa only [view.regionEq] using read_eq alloc t actual binding initial i bound

structure Forward (p : Parameters) (views : ForwardViews p) (alloc : RegionName → Allocation) where
  logits : VeriTile.Torch.FP32.Tensor
  loss : VeriTile.Torch.FP32.Tensor
  lse : VeriTile.Torch.FP32.Tensor
  zloss : VeriTile.Torch.FP32.Tensor
  logitsBinding : Binding alloc views.logits logits
  lossBinding : Binding alloc views.loss loss
  lseBinding : Binding alloc views.lse lse
  zlossBinding : Binding alloc views.zloss zloss

structure Backward (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation) where
  logits : VeriTile.Torch.FP32.Tensor
  lse : VeriTile.Torch.FP32.Tensor
  upstream : VeriTile.Torch.FP32.Tensor
  gradient : VeriTile.Torch.FP32.Tensor
  logitsBinding : Binding alloc views.logits logits
  lseBinding : Binding alloc views.lse lse
  upstreamBinding : Binding alloc views.upstream upstream
  gradientBinding : Binding alloc views.gradient gradient

#axiomsClean physical_address
#axiomsClean initial_word
#axiomsClean read_eq
#axiomsClean scalar_numel
#specNonCircular Binding avoiding [forwardSyntax,backwardSyntax,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelBinding
