import bench.tritonbench_g.fast_rms_layernorm.Contract
import bench.tritonbench_g.fast_rms_layernorm.Proofs.KernelStorage
import VeriTile.Torch.Tensor

/-! Tensor views and their physical reads at kernel entry. These are ordinary
representation laws: no premise mentions a kernel run or its desired result.
The contiguous read law permits arbitrary singleton strides and storage offsets.
Backward's dx tensor is the already existing private upstream clone. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.KernelTensor
open VeriTile.Triton KernelStorage
noncomputable section

def inputs : Phase → List RegionName
  | .forward => ["x","w"]
  | .backward => ["x","w","r","dx"]

theorem input_member (phase : Phase) (role : RegionName) (hr : role∈inputs phase) :
    role∈buffers phase := by cases phase <;> simp_all [inputs,buffers] <;> tauto

/-- Observations of the actual Tensor handles used by the launches. No stride
equality is imposed: default contiguity ignores empty and singleton strides. -/
structure Binding (h : Call) (phase : Phase) (s : Placement h phase)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) : Prop where
  grid : h.rows≤2147483647
  settings : h.block=Settings.block h.cols
  shape : ∃ leading,(tensors "x").shape=leading++[h.cols] ∧ h.rows=leading.prod
  cuda : ∀ role, role∈inputs phase → (tensors role).device.kind="cuda"
  device : ∀ role, role∈inputs phase → (tensors role).device=(tensors "x").device
  count : ∀ role, role∈inputs phase → (tensors role).numel=cells h role
  dtype : ∀ role, role∈inputs phase → (tensors role).dtype=.floating .fp32
  layout : ∀ role, role∈inputs phase → (tensors role).layout=.strided
  contiguous : ∀ role, role∈inputs phase → (tensors role).isContiguous=Bool.true
  noNeg : ∀ role, role∈inputs phase → (tensors role).isNeg=Bool.false
  pointer : ∀ role (hr : role∈inputs phase), 0<cells h role →
    (tensors role).dataPtr.toNat=(s.allocation role (input_member phase role hr)).buffer.base

/-- Decode the four physical input bytes at the Tensor's data pointer. -/
def ReadLaw {β γ δ : Type} (phase : Phase)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (heap : Nat → β)
    (decode : (Fin 4 → β) → γ) (observe : γ → δ) (logical : RegionName → Nat → δ) : Prop :=
  ∀ role, role∈inputs phase → ∀ off,off<(tensors role).numel →
    observe (decode (fun byte => heap ((tensors role).dataPtr.toNat+off*4+byte.val)))=logical role off

/-- The interpreter starts from these physical reads, including its scratch
regions. No input value is supplied by a host callback. -/
def lift {β γ : Type} {h : Call} {phase : Phase} (s : Placement h phase)
    (heap : Nat → β) (decode : (Fin 4 → β) → γ) (other : γ) : RegionName × Nat → γ :=
  fun a => if hr : a.1∈buffers phase then
    decode (fun byte => heap (((BitVec.ofNat 64 (s.allocation a.1 hr).buffer.base +
      BitVec.ofNat 64 a.2 * BitVec.ofNat 64 4).toNat)+byte.val)) else other

theorem logical_read {β γ δ : Type} (h : Call) (phase : Phase) (s : Placement h phase)
    (tensors : RegionName → VeriTile.Torch.TensorMetadata) (binding : Binding h phase s tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → γ) (observe : γ → δ)
    (logical : RegionName → Nat → δ) (reads : ReadLaw phase tensors heap decode observe logical)
    (other : γ) (role : RegionName) (hr : role∈inputs phase) (off : Nat) (inside : off<cells h role) :
    observe (lift s heap decode other (role,off))=logical role off := by
  have member:=input_member phase role hr
  have ptr:=binding.pointer role hr (by omega)
  have count:=binding.count role hr
  have native := (NativeGrid.physical_bytes (cells h role) (s.allocation role member) off inside).1
  simp only [lift,dif_pos member,Prod.fst,Prod.snd,native]
  simpa only [ptr] using reads role hr off (by omega)

def rowValues {δ : Type} (h : Call) (logical : RegionName → Nat → δ)
    (role : RegionName) (row : Fin h.rows) (j : Fin h.cols) : δ := logical role (row.val*h.cols+j.val)

def weightValues {δ : Type} (h : Call) (logical : RegionName → Nat → δ) (j : Fin h.cols) : δ :=
  logical "w" j.val

#axiomsClean logical_read
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.KernelTensor
