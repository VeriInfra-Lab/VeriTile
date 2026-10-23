import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.RawMemory
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.TensorBridge
open VeriTile Triton
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Ordinary observation of a real PyTorch FP32 Tensor. These fields concern
only object/metadata/storage observations. They contain no source computation,
launch, numerical result, or logical output equality. The Boolean floating
classification means exactly FP32 on this selected checked path. -/
structure Observation (raw : Tensor) (tensor : Torch.FP32.Tensor) : Prop where
  laws : Torch.FP32.TensorLaws tensor
  object : tensor.object=raw.object
  shape : tensor.metadata.shape=raw.shape
  dtype : tensor.metadata.dtype=if raw.floating then .floating .fp32 else .other "int32"
  strided : tensor.metadata.isStrided=raw.strided
  cuda : tensor.metadata.isCuda=raw.cuda
  device : tensor.metadata.device.index=raw.device
  contiguous : tensor.metadata.isContiguous=raw.contiguous
  negative : tensor.metadata.isNeg=raw.negative
  pointer : tensor.dataPtr.toNat=raw.pointer
  storageBase : tensor.storage.base=raw.storageBase
  storageBytes : 4*tensor.storage.elements=raw.storageBytes
  storageOffset : tensor.storageOffset=raw.storageOffset

def logicalWord (api : API (BitVec 32) Store Byte) (memory : Store)
    (tensor : Torch.FP32.Tensor) (binding : Observation raw tensor) (i : Nat) : BitVec 32 :=
  (Torch.FP32.logicalValue (fun address=>⟨api.read memory address⟩) tensor binding.laws i).bits

/-- Logical Tensor readback uses the actual allocated storage through shared
TensorLaws. Bounds come from the checked output shape, and the address is the
same observed pointer used by the source's physical store projection. -/
theorem logical_read (api : API (BitVec 32) Store Byte) (memory : Store)
    (accepted : Admission c request values) (tensor : Torch.FP32.Tensor)
    (binding : Observation (request.tensor .output) tensor) (q : Raw.Coord c) :
    logicalWord api memory tensor binding (Raw.offset q)=api.read memory (Raw.address request q) := by
  obtain ⟨shape,dtype,cuda,device,contiguous,negative,conjugate,strided⟩:=accepted.schema.header .output
  have hc : tensor.metadata.isContiguous=Bool.true := binding.contiguous.trans contiguous
  have hn : tensor.metadata.isNeg=Bool.false := binding.negative.trans negative
  have hshape : tensor.metadata.shape=[c.Z,c.H,c.T,c.BD] := binding.shape.trans shape
  have bound : Raw.offset q<tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,hshape,tensorSize,Nat.mul_assoc] using Raw.offset_bound q
  unfold logicalWord Torch.FP32.logicalValue
  simp only [hn,Bool.false_eq_true,ite_false,binding.laws.contiguous hc _ bound]
  change api.read memory (tensor.storage.base+4*(tensor.storageOffset+Raw.offset q))=_
  rw [binding.storageBase,binding.storageOffset]
  apply congrArg (api.read memory)
  unfold Raw.address
  rw [(accepted.storage .output).1]
  omega

#axiomsClean logical_read
#specNonCircular Observation avoiding [Raw.execute,Raw.dispatch,Physical.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.TensorBridge
