import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Contract
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.NativeBinding
import VeriTile.Triton.Semantics.State

/-! Kernel input metadata binds to actual scalar coordinates and pointers.
There is no wrapper run, allocator or output equation in this interface. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Admission
open Kernels Tensors Execution TensorLaws
open VeriTile.Triton ByteMemory
set_option autoImplicit false

def dtype : Route → VeriTile.Torch.DType
  | .real32 => .floating .fp32
  | .complex64 => .other "complex64"

theorem dtype_injective {a b : Route} (h : dtype a = dtype b) : a = b := by
  cases a <;> cases b <;> simp_all [dtype]

/-- The descriptor binds exactly the original Tensor. The pointer equation
is required only for nonempty tensors; no unused empty-view offset is checked.
Dimensions are ordinary signed-64 Torch sizes and backing capacities are
part of View. Contiguity is interpreted through the actual scalar coordinates. -/
structure Binding (m : Contract.Metadata) (tensor : Tensor) : Prop where
  dtype : m.dtype = Admission.dtype tensor.route
  device : m.device = tensor.device
  shape : m.shape = tensor.shape
  contiguous : m.isContiguous = tensor.contiguous
  negative : m.isNeg = tensor.negative
  conjugated : m.isConj = tensor.conjugated
  size : tensor.view.words = tensor.route.components * tensor.count
  coordinate : tensor.contiguous = Bool.true → ∀ q, q < tensor.count →
    tensor.coordinate q = tensor.route.components * q
  pointer : 0 < tensor.count → m.dataPtr.toNat = tensor.view.pointer
  metadata : NativeBinding.Metadata tensor

def dimensions (m : Contract.Metadata) : Dimensions :=
  ⟨m.shape.getD 0 0, m.shape.getD 1 0, m.shape.getD 2 0⟩

/-- Bind each region to its tensor's view of one shared initial byte heap.
Different regions may have identical or partially overlapping physical views.
This constructs a read snapshot, not a disjoint allocation or a host execution.
`decode` supplies the mathematical interpretation of a stored component. -/
noncomputable def bindReadViews {α : Type} (codec : Codec α) (decode : α → ℝ)
    (heap : Heap) (views : RegionName → Tensor) (scratch : BlockState) : BlockState :=
  { scratch with mem := fun r o => MemCell.real (decode (load codec heap ((views r).view.address o))) }

/-- The independent regional snapshot reads the caller's logical components.
No assumption separates one read-only input from another. -/
theorem bindReadViews_logical {α : Type} (codec : Codec α) (decode : α → ℝ)
    (signFlip : α → α) (heap : Heap) (views : RegionName → Tensor) (scratch : BlockState)
    (r : RegionName) (dense : Dense (views r)) (resolved : Resolved (views r))
    (q : Nat) (hq : q < (views r).count) (k : Nat) :
    (bindReadViews codec decode heap views scratch).readMem r ((views r).route.components * q + k) =
      decode (logicalWord codec signFlip (views r) heap q k) := by
  rw [resolved_word codec signFlip (views r) heap resolved dense q hq k]
  rfl

theorem shape_of_rank (m : Contract.Metadata) (rank : m.shape.length = 3) :
    m.shape = [(dimensions m).length, (dimensions m).batch, (dimensions m).dim] := by
  rcases m with ⟨⟨dtype,layout,device,shape,contiguous,neg,ptr⟩,conj⟩
  dsimp only at rank ⊢
  cases shape with
  | nil => simp at rank
  | cons l tail => cases tail with
    | nil => simp at rank
    | cons b tail => cases tail with
      | nil => simp at rank
      | cons d tail =>
        have empty : tail = [] := by simpa using rank
        subst tail
        rfl

theorem dense (m : Contract.Metadata) (tensor : Tensor) (binding : Binding m tensor)
    (admitted : Contract.TensorContract Bool.true m) : Dense tensor := by
  have hc : tensor.contiguous = Bool.true := binding.contiguous.symm.trans (admitted.dense rfl)
  exact ⟨hc, binding.size, binding.coordinate hc⟩

theorem resolved (c : Bool) (m : Contract.Metadata) (tensor : Tensor) (binding : Binding m tensor)
    (admitted : Contract.TensorContract c m) : Resolved tensor :=
  ⟨binding.negative.symm.trans admitted.resolved.1, binding.conjugated.symm.trans admitted.resolved.2⟩

theorem aligned (c : Bool) (m : Contract.Metadata) (tensor : Tensor) (binding : Binding m tensor)
    (admitted : Contract.TensorContract c m) (nonempty : 0 < tensor.count) :
    tensor.view.pointer % 4 = 0 := by
  rw [← binding.pointer nonempty]
  apply admitted.alignment
  simpa [VeriTile.Torch.TensorMetadata.numel, Tensor.count, binding.shape] using nonempty

theorem layouts (smeta xmeta lmeta : Contract.Metadata) (s x transition : Tensor)
    (h : Contract.InputContract smeta xmeta lmeta)
    (sb : Binding smeta s) (xb : Binding xmeta x) (lb : Binding lmeta transition) :
    Layout x.route [(dimensions xmeta).batch, (dimensions xmeta).dim] s ∧
    Layout x.route [(dimensions xmeta).length, (dimensions xmeta).batch, (dimensions xmeta).dim] x ∧
    Layout x.route [(dimensions xmeta).dim] transition := by
  have sx : s.route = x.route := dtype_injective (sb.dtype.symm.trans (h.dtype.1.trans xb.dtype))
  have lx : transition.route = x.route := dtype_injective (lb.dtype.symm.trans (h.dtype.2.symm.trans xb.dtype))
  have xs := shape_of_rank xmeta h.rank
  refine ⟨⟨sx, ?_, dense smeta s sb h.initial⟩,
    ⟨rfl, xb.shape.symm.trans xs, dense xmeta x xb h.input⟩,
    ⟨lx, ?_, dense lmeta transition lb h.parameter⟩⟩
  · simpa [xs] using sb.shape.symm.trans h.initialShape
  · simpa [xs] using lb.shape.symm.trans h.parameterShape

/-- Native integer and loop bounds follow from the very same admitted tensor
shapes and ordinary allocation/metadata laws, including zero time. -/
theorem native_shape (smeta xmeta lmeta : Contract.Metadata) (s x transition : Tensor)
    (h : Contract.InputContract smeta xmeta lmeta)
    (sb : Binding smeta s) (xb : Binding xmeta x) (lb : Binding lmeta transition) :
    NativeBinding.Shape x.route (dimensions xmeta) := by
  obtain ⟨sl,xl,_⟩ := layouts smeta xmeta lmeta s x transition h sb xb lb
  exact NativeBinding.from_tensors x.route (dimensions xmeta) s x sl.dtype xl.dtype
    sl.shape xl.shape sl.dense xl.dense sb.metadata xb.metadata

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Admission
