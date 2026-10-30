import bench.tritonbench_g.parallel_retention_attention.Proofs.Contract
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Checked
noncomputable section
open VeriTile.Torch

/-- Observations before the original contiguous copies. Exact Tensor class and
conjugate-bit observations supplement the shared ordinary metadata interface. -/
structure Tensor where
  metadata : VeriTile.Torch.TensorMetadata
  exactTensor : Bool
  conjugate : Bool

def Tensor.Valid (x : Tensor) : Prop :=
  x.exactTensor=true ∧ x.metadata.layout=.strided ∧ x.metadata.dtype=.floating .fp32 ∧
  x.metadata.device.kind="cuda" ∧ x.metadata.isNeg=false ∧ x.conjugate=false

structure Inputs where
  query : Tensor
  key : Tensor
  value : Tensor

/-- Both feature tiles and every source integer guard occur here verbatim.
The source retains its original input copies; arbitrary-stride kernel execution
is not an admitted interpretation. Empty tensors have no pointer premise. -/
structure InputContract (s : Shape) (x : Inputs) : Prop where
  query : x.query.Valid
  key : x.key.Valid
  value : x.value.Valid
  queryShape : x.query.metadata.shape=[s.B,s.H,s.T,s.DK]
  keyShape : x.key.metadata.shape=x.query.metadata.shape
  valueShape : x.value.metadata.shape=[s.B,s.H,s.T,s.DV]
  devices : x.key.metadata.device=x.query.metadata.device ∧ x.value.metadata.device=x.query.metadata.device
  geometry : ShapeContract s

/-- Literal numeric checks from `_checked_geometry`, with the selected tiles. -/
def GeometryGuards (s : Shape) : Prop :=
  s.BK=featureWidth s.DK ∧ s.BV=featureWidth s.DV ∧ 0<s.DK ∧ 0<s.DV ∧
  blocks s.T 128*128+128<2^31 ∧ blocks s.DK s.BK*s.BK+s.BK<2^31 ∧
  blocks s.DV s.BV*s.BV+s.BV<2^31 ∧ blocks s.DK s.BK*blocks s.DV s.BV<2^31 ∧
  blocks s.T 64<65536 ∧ s.B*s.H<65536 ∧
  ((blocks s.DV s.BV*s.B*s.H+1)*s.T+blocks s.T 128*128+128)*s.DK+
    blocks s.DK s.BK*s.BK+s.BK<2^61 ∧
  ((blocks s.DK s.BK*s.B*s.H+1)*s.T+blocks s.T 128*128+128)*s.DV+
    blocks s.DV s.BV*s.BV+s.BV<2^61

theorem geometry_iff (s : Shape) : GeometryGuards s ↔ ShapeContract s := by
  constructor
  · rintro ⟨kt,vt,dk,dv,tp,kp,vp,gx,gy,gz,fk,fv⟩
    refine ⟨kt,vt,⟨dk,dv,?_,?_,?_,?_,tp,kp,vp,gx,gy,gz,fk,fv⟩⟩
    · rw [kt];exact featureWidth_positive s.DK
    · rw [vt];exact featureWidth_positive s.DV
    · rw [kt];exact featureWidth_tiles s.DK
    · rw [vt];exact featureWidth_tiles s.DV
  · intro h
    exact ⟨h.keyTile,h.valueTile,h.indices.dk_pos,h.indices.dv_pos,
      h.indices.time_padded,h.indices.key_padded,h.indices.value_padded,h.indices.grid_x,
      h.indices.grid_y,h.indices.grid_z,h.indices.formed_key,h.indices.formed_value⟩

/-- Literal metadata/numeric admission; no classical decision procedure or
kernel output is used to decide the predicate. -/
instance (x : Tensor) : Decidable x.Valid := by unfold Tensor.Valid;infer_instance
instance (s : Shape) : Decidable (GeometryGuards s) := by unfold GeometryGuards;infer_instance

def admit (s : Shape) (x : Inputs) : Bool :=
  decide (x.query.Valid ∧ x.key.Valid ∧ x.value.Valid ∧
    x.query.metadata.shape=[s.B,s.H,s.T,s.DK] ∧ x.key.metadata.shape=x.query.metadata.shape ∧
    x.value.metadata.shape=[s.B,s.H,s.T,s.DV] ∧
    (x.key.metadata.device=x.query.metadata.device ∧ x.value.metadata.device=x.query.metadata.device) ∧
    GeometryGuards s)

theorem admit_iff (s : Shape) (x : Inputs) : admit s x=true ↔ InputContract s x := by
  simp only [admit,decide_eq_true_eq]
  constructor
  · rintro ⟨q,k,v,qs,ks,vs,devices,geom⟩
    exact ⟨q,k,v,qs,ks,vs,devices,(geometry_iff s).mp geom⟩
  · intro h
    exact ⟨h.query,h.key,h.value,h.queryShape,h.keyShape,h.valueShape,h.devices,(geometry_iff s).mpr h.geometry⟩

def formedKey (s : Shape) :=
  ((blocks s.DV s.BV*s.B*s.H+1)*s.T+blocks s.T 128*128+128)*s.DK+blocks s.DK s.BK*s.BK+s.BK
def formedValue (s : Shape) :=
  ((blocks s.DK s.BK*s.B*s.H+1)*s.T+blocks s.T 128*128+128)*s.DV+blocks s.DV s.BV*s.BV+s.BV

/-- Ordinary metadata observations of the actual contiguous-copy results.
The pointer guards are checked after those necessary copies, so their fresh
pointers, or retained already-contiguous pointers, are the ones consumed. -/
structure Prepared (s : Shape) (x : Inputs) (b : Bases) where
  query : VeriTile.Torch.TensorMetadata
  key : VeriTile.Torch.TensorMetadata
  value : VeriTile.Torch.TensorMetadata
  descriptors : query.shape=x.query.metadata.shape ∧ key.shape=x.key.metadata.shape ∧
    value.shape=x.value.metadata.shape ∧ query.device=x.query.metadata.device ∧
    key.device=x.key.metadata.device ∧ value.device=x.value.metadata.device
  contiguous : query.isContiguous=true ∧ key.isContiguous=true ∧ value.isContiguous=true
  fp32 : query.dtype=.floating .fp32 ∧ key.dtype=.floating .fp32 ∧ value.dtype=.floating .fp32
  pointer : s.B*s.H*s.T>0 →
    query.dataPtr.toNat=4*b.q ∧ key.dataPtr.toNat=4*b.k ∧ value.dataPtr.toNat=4*b.v
  checked : s.B*s.H*s.T>0 → PointerDomain query.dataPtr.toNat (formedKey s) ∧
    PointerDomain key.dataPtr.toNat (formedKey s) ∧ PointerDomain value.dataPtr.toNat (formedValue s)

theorem prepared_key_address (s : Shape) (x : Inputs) (h : InputContract s x)
    (b : Bases) (prepared : Prepared s x b) (active : s.B*s.H*s.T>0)
    (off : Nat) (bound : off≤formedKey s) :
    4*b.q+4*off<2^64 ∧ 4*off<2^63 := by
  have ptr := (prepared.pointer active).1
  have room := formed_address prepared.query.dataPtr.toNat (formedKey s) off (prepared.checked active).1 bound
  have signed := signed64_byte_offsets (formedKey s) off h.geometry.indices.formed_key bound
  omega

/-- Successful checked scratch allocations: the Python wrapper checks these
actual fresh pointers before either source launch. -/
structure ScratchPointers (s : Shape) (b : Bases) (fl : ForwardLayout s b) (bl : BackwardLayout s b) : Prop where
  forward : s.B*s.H*s.T>0 → PointerDomain (4*fl.scratch) (formedValue s)
  query : s.B*s.H*s.T>0 → PointerDomain (4*bl.scratch .query) (formedKey s)
  key : s.B*s.H*s.T>0 → PointerDomain (4*bl.scratch .key) (formedKey s)
  value : s.B*s.H*s.T>0 → PointerDomain (4*bl.scratch .value) (formedValue s)

theorem scratch_addresses (s : Shape) (b : Bases) (fl : ForwardLayout s b) (bl : BackwardLayout s b)
    (pointers : ScratchPointers s b fl bl) (active : s.B*s.H*s.T>0)
    (k v : Nat) (hk : k≤formedKey s) (hv : v≤formedValue s) :
    4*fl.scratch+4*v<2^64 ∧ 4*bl.scratch .query+4*k<2^64 ∧
    4*bl.scratch .key+4*k<2^64 ∧ 4*bl.scratch .value+4*v<2^64 := by
  have f := formed_address _ _ v (pointers.forward active) hv
  have q := formed_address _ _ k (pointers.query active) hk
  have dk := formed_address _ _ k (pointers.key active) hk
  have dv := formed_address _ _ v (pointers.value active) hv
  omega

#axiomsClean scratch_addresses
#axiomsClean geometry_iff
#axiomsClean admit_iff
#axiomsClean prepared_key_address
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Checked
