import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Contract
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.ByteCodec
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton Contract
noncomputable section

/-- Actual shared FP32 Tensor, ordinary API facts and the full untyped backing.
No relation between distinct StorageImpl identities and physical spans occurs. -/
structure FP32Binding (info : TensorInfo) where
  tensor : Torch.FP32.Tensor
  laws : Torch.FP32.TensorLaws tensor
  metadata : tensor.metadata=info.toTensorMetadata
  object : tensor.object=info.object
  storage : tensor.storage.object=info.backing
  offset : tensor.storageOffset=info.storageOffset
  base : tensor.storage.base=info.storageBase
  extent : tensor.storage.elements=info.storageBytes/4
  fullSpan : info.storageBase+info.storageBytes≤2^64

/-- The repository has no shared half/integer Tensor carrier. These ordinary
API laws bind its descriptor and logical element map to its physical bytes. -/
structure RawBinding (info : TensorInfo) (bytes : Nat) where
  physicalIndex : Nat → Nat
  span : info.storageBase+info.storageBytes≤2^64
  bounds : ∀ j,j< info.numel → bytes*(physicalIndex j+1)≤ info.storageBytes
  contiguous : info.isContiguous=Bool.true → ∀ j,j< info.numel → physicalIndex j=info.storageOffset+j
  pointer : 0< info.numel → info.dataPtr.toNat=info.storageBase+bytes*info.storageOffset

def indexBytes (i : TensorInfo) : Nat := if i.dtype=.other "int64" then 8 else 4

structure Request where
  x : TensorInfo
  w : TensorInfo
  y : TensorInfo
  i : TensorInfo
  scalars : Scalars
  input : FP32Binding x
  weights : RawBinding w 2
  output : FP32Binding y
  metadata : RawBinding i (indexBytes i)

def half (heap : ByteMemory) (address : Nat) : Word :=
  (heap (address+1) ++ heap address).zeroExtend 32

def indices (r : Request) (heap : ByteMemory) (b : Nat) : Int :=
  let p := r.i.dataPtr.toNat+indexBytes r.i*b
  if r.i.dtype=.other "int64" then (load32 heap (p+4) ++ load32 heap p).toInt
  else (load32 heap p).toInt

def Admitted (r : Request) (heap : ByteMemory) : Prop :=
  Contract.Admitted r.x r.w r.y r.i r.scalars (indices r heap)

def checked (r : Request) (heap : ByteMemory) (h : Admitted r heap) :
    InputContract r.x r.w r.y r.i r.scalars (indices r heap) :=
  (python_checks_iff ..).mp h

def call (r : Request) (heap : ByteMemory) (h : Admitted r heap) : Call :=
  Contract.call r.x r.w r.y r.i r.scalars (indices r heap) h

def logical {α : Type} (read : Nat → α) (t : Torch.FP32.Tensor) (j : Nat) : α :=
  read (t.storage.address (t.physicalIndex j))

theorem pointer_address (a : FP32Binding info) (hc : info.isContiguous=Bool.true)
    (j : Nat) (hj : j< info.numel) :
    a.tensor.storage.address (a.tensor.physicalIndex j)=info.dataPtr.toNat+4*j := by
  have cap := Torch.FP32.contiguous_capacity a.tensor a.laws (by simpa [a.metadata] using hc) (by simpa [a.metadata] using (show 0< info.numel by omega))
  have ptr := a.laws.pointer (by simpa [a.metadata] using (show 0< info.numel by omega))
  have span := a.tensor.storage.spanFits
  have bound : a.tensor.storage.address a.tensor.storageOffset<2^64 := by
    simp only [Torch.FP32.Storage.address]
    have : a.tensor.metadata.numel=info.numel := congrArg Torch.TensorMetadata.numel a.metadata
    omega
  rw [a.laws.contiguous (by simpa [a.metadata] using hc) j (by simpa [a.metadata] using hj)]
  have ptrNat := congrArg BitVec.toNat ptr
  simp only [Torch.FP32.Tensor.dataPtr,a.metadata,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at ptrNat
  rw [ptrNat]
  simp [Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem logical_direct {α : Type} (a : FP32Binding info) (read : Nat → α)
    (hc : info.isContiguous=Bool.true) (j : Nat) (hj : j< info.numel) :
    logical read a.tensor j=read (info.dataPtr.toNat+4*j) := by
  rw [logical,pointer_address a hc j hj]

theorem fp32_logical (a : FP32Binding info) (heap : ByteMemory) (hn : info.isNeg=Bool.false)
    (j : Nat) : Torch.FP32.logicalValue (fun p => ⟨load32 heap p⟩) a.tensor a.laws j=
      ⟨logical (load32 heap) a.tensor j⟩ := by
  simp [Torch.FP32.logicalValue,a.metadata,hn,logical]

theorem fp32_span (a : FP32Binding info) (hc : info.isContiguous=Bool.true)
    (j : Nat) (hj : j< info.numel) (byte : Fin 4) :
    info.storageBase≤ info.dataPtr.toNat+4*j+byte.val ∧
      info.dataPtr.toNat+4*j+byte.val< info.storageBase+info.storageBytes := by
  rw [←pointer_address a hc j hj]
  have bound := a.laws.inBounds j (by simpa [a.metadata] using hj)
  rw [a.extent] at bound
  have div := Nat.mod_add_div info.storageBytes 4
  have rem := Nat.mod_lt info.storageBytes (by decide : 0<4)
  simp only [Torch.FP32.Storage.address,a.base]
  have := byte.isLt
  omega

theorem raw_pointer (a : RawBinding info bytes) (hc : info.isContiguous=Bool.true)
    (j : Nat) (hj : j< info.numel) :
    info.storageBase+bytes*a.physicalIndex j=info.dataPtr.toNat+bytes*j := by
  rw [a.contiguous hc j hj,a.pointer (by omega)]
  simp [Nat.mul_add,Nat.add_assoc]

theorem raw_span (a : RawBinding info bytes) (hc : info.isContiguous=Bool.true)
    (j : Nat) (hj : j< info.numel) (b : Nat) (hb : b<bytes) :
    info.storageBase≤ info.dataPtr.toNat+bytes*j+b ∧
      info.dataPtr.toNat+bytes*j+b< info.storageBase+info.storageBytes := by
  rw [←raw_pointer a hc j hj]
  have bound := a.bounds j hj
  simp only [Nat.mul_add,Nat.mul_one] at bound
  omega

#axiomsClean pointer_address
#axiomsClean fp32_span
#axiomsClean raw_pointer
#axiomsClean raw_span
#specNonCircular Admitted avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
