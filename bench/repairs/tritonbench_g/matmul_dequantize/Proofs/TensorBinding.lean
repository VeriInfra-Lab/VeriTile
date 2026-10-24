import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.ByteBridge
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TensorBinding
open VeriTile.Triton Host
noncomputable section

/-- Actual half/int32 Tensor ABI, and the shared FP32 Tensor when selected.
The storage name is a stable name for the same raw caller allocation. -/
structure Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storageName : RegionName
  storageObject : Nat
  storageBase : Nat
  storageBytes : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat
  isConj : Bool
  fp32 : Option VeriTile.Torch.FP32.Tensor

def dtype : StorageDType → VeriTile.Torch.DType
  | .fp16 => .other "float16" | .fp32 => .floating .fp32 | .int32 => .other "int32"

structure Binding (d : Host.Tensor) (t : Tensor) : Prop where
  object : t.object=d.identity
  shape : t.metadata.shape=[d.rows,d.columns]
  dtype : t.metadata.dtype=TensorBinding.dtype d.dtype
  layout : t.metadata.layout=.strided
  device : t.metadata.device=⟨"cuda",d.device⟩
  contiguous : t.metadata.isContiguous=Bool.true
  negative : t.metadata.isNeg=Bool.false
  conjugate : t.isConj=Bool.false
  name : t.storageName=d.region
  base : t.storageBase=d.base
  bytes : t.storageBytes=Host.bytes d*d.allocation
  offset : t.storageOffset=d.storageOffset
  pointer : t.metadata.dataPtr=BitVec.ofNat 64 (d.base+Host.bytes d*d.storageOffset)
  coordinates : ∀ r n,r<d.rows → n<d.columns →
    t.physicalIndex (r*d.columns+n)=d.storageOffset+r*d.rowStride+n*d.columnStride
  /-- Ordinary FP32 object/storage binding, with no numerical output law. -/
  shared : d.dtype=.fp32 → ∃ x : VeriTile.Torch.FP32.Tensor,t.fp32=some x ∧
    VeriTile.Torch.FP32.TensorLaws x ∧ x.object=t.object ∧ x.metadata=t.metadata ∧
    x.storage.object=t.storageObject ∧ x.storage.base=t.storageBase ∧
    4*x.storage.elements=t.storageBytes ∧ x.storageOffset=t.storageOffset ∧
    x.physicalIndex=t.physicalIndex

/-- Unused descriptors in the historical six-slot source interface are ghost
metadata only. No corresponding caller Tensor, allocation or device action is
required; only the route's actual arguments/scratch/results are bound. -/
def live (a : Arguments) (route : Route) : List Host.Tensor :=
  if route=.dequant16 ∨ route=.dequant32 then [a.q,a.s,a.z,a.d]
  else if route=.s1 then tensors a else [a.a,a.q,a.s,a.z,a.c]

def output (a : Arguments) (route : Route) : Host.Tensor :=
  if route=.dequant16 ∨ route=.dequant32 then a.d else a.c

/-- The dequantization helper always allocates D, including when the outer
s1 caller supplied C. The other APIs allocate exactly when C is absent. -/
def allocated (a : Arguments) (route : Route) : Bool :=
  if route=.dequant16 ∨ route=.dequant32 then Bool.true else !a.supplied

structure Tensors (a : Arguments) (route : Route) where
  get : RegionName → Tensor
  bound : ∀ d∈live a route,Binding d (get d.region)
  storageIdentity : ∀ r s,(get r).storageObject=(get s).storageObject → r=s

 theorem output_bound (a : Arguments) (route : Route) (t : Tensors a route) :
    Binding (output a route) (t.get (output a route).region) := by
  apply t.bound
  cases route <;> simp [output,live,tensors]

structure Result where
  memory : ByteBridge.Memory
  output : Tensor
  returned : Option Tensor
  allocated : Bool
  scratchReleased : Bool

def returned (a : Arguments) (route : Route) (t : Tensors a route) : Option Tensor :=
  if route=.gptq ∧ a.supplied then none else some (t.get (output a route).region)

#axiomsClean output_bound
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.TensorBinding
