import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.IO
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Tensor

/-! Optional physical-byte application of the typed regional IO in
`KernelOnly/IO.lean`. This view binds caller Tensor pointers, metadata reads,
and the allocated INT64 scheduler buffers to one initial heap. Host allocation
and launch realization remain input ABI bindings, not kernel conclusions. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling
open VeriTile.Triton
noncomputable section

namespace SourceTensor

/-- Little-endian signed integer load; no floating-point conversion. -/
def readSigned (bits : Nat) (m : Nat → Byte) (address : Nat) : Int :=
  (BitVec.ofNat bits ((List.range (bits / 8)).foldl
    (fun value b => value + (m (address+b)).toNat * 256^b) 0)).toInt

/-- Metadata tensors may be strided and carry Torch's negative-view bit.
`data_ptr` already includes the storage offset. Strides are in elements. -/
def metadataValue (strides : TensorDescriptor × Nat → Int)
    (m : Nat → Byte) (t : TensorDescriptor) (row col : Nat) : Int :=
  let address := (t.metadata.dataPtr.toNat : Int) +
    4*((row : Int)*strides (t,0)+(col : Int)*strides (t,1))
  let value := readSigned 32 m address.toNat
  if t.metadata.isNeg then -value else value

/-- Actual queried strides are indexed by their Tensor descriptor and axis.
Only active page entries are bound: unused negative entries remain allowed.
These conditions concern input bytes exclusively. -/
structure Metadata (t : TensorInputs) (r : Request) (m : Nat → Byte) where
  strides : TensorDescriptor × Nat → Int
  storage : ∀ tensor ∈ [t.start,t.count,t.total,t.table], ∀ (row col : Nat),
    row < tensor.metadata.shape[0]?.getD 0 →
    col < (if tensor = t.table then r.tableWidth else 1) →
    let address := (tensor.metadata.dataPtr.toNat : Int) +
      4*((row : Int)*strides (tensor,0)+(col : Int)*strides (tensor,1))
    (tensor.storageBase : Int) ≤ address ∧ address+4 ≤ tensor.storageBase+tensor.storageBytes
  start : ∀ b, b < r.batches → metadataValue strides m t.start b 0 = (r.start b : Int)
  count : ∀ b, b < r.batches → metadataValue strides m t.count b 0 = (r.count b : Int)
  total : ∀ b, b < r.batches → metadataValue strides m t.total b 0 = (r.total b : Int)
  page : ∀ b i, b < r.batches → i < r.count b →
    metadataValue strides m t.table b ((r.history b+i)/r.block) =
      (r.page b ((r.history b+i)/r.block) : Int)

/-- Ordinary launch bindings for the scheduler buffers allocated by the host.
The buffers' signed loads are linked to the validated request, without any
output, successful-execution, or numerical-agreement premise. -/
structure Schedule (t : TensorInputs) (r : Request) (m : Nat → Byte) where
  metadata : Metadata t r m
  sourcePointer : Nat
  destinationPointer : Nat
  index : Job → Nat
  enumerates : ∀ j ∈ jobs r, (jobs r)[index j]? = some j
  sourceRead : ∀ j ∈ jobs r,
    readSigned 64 m (sourcePointer + 8*index j) = (sourceRow r j : Int)
  destinationRead : ∀ j ∈ jobs r,
    readSigned 64 m (destinationPointer + 8*index j) = (destinationRow r j : Int)

def row {α : Type} (decode : (Fin 4 → Byte) → α) (t : TensorInputs) (r : Request)
    (m : Nat → Byte) (j : Job) (key : Bool) (i : Nat) : α :=
  decode (fun b => Tensor.byte (if key then t.key else t.value) m
    (sourceRow r j * (if key then r.dk else r.dv) + i) b)

end SourceTensor

namespace WholeBacking

/-- Typed regional loads read the same caller bytes. Unknown regions retain
the supplied scratch state; no active input is synthesized from a formula. -/
def memory {α : Type} (decode : (Fin 4 → Byte) → α) (t : TensorInputs) (r : Request)
    (m : Nat → Byte) (schedule : SourceTensor.Schedule t r m)
    (scratch : FP.Structural.State α) : RegionName → Nat → FP.Structural.Cell α :=
  fun region i =>
    if region = "src" then .mk .int (SourceTensor.readSigned 64 m (schedule.sourcePointer+8*i))
    else if region = "dst" then .mk .int (SourceTensor.readSigned 64 m (schedule.destinationPointer+8*i))
    else if region = "key" then .mk .real (decode (fun b => Tensor.byte t.key m i b))
    else if region = "value" then .mk .real (decode (fun b => Tensor.byte t.value m i b))
    else if region = "keyCache" then
      if r.policy = .copy then .mk .real (decode (fun b => Tensor.byte t.keyCache m i b))
      else .mk .nat ((Tensor.byte t.keyCache m i 0).toNat)
    else if region = "valueCache" then
      if r.policy = .copy then .mk .real (decode (fun b => Tensor.byte t.valueCache m i b))
      else .mk .nat ((Tensor.byte t.valueCache m i 0).toNat)
    else if region = "keyMeta" then .mk .real (decode (fun b => Tensor.byte (Tensor.output t .keyMeta) m i b))
    else if region = "valueMeta" then .mk .real (decode (fun b => Tensor.byte (Tensor.output t .valueMeta) m i b))
    else scratch.mem region i

end WholeBacking

namespace SourceTensor

def initial {α : Type} (decode : (Fin 4 → Byte) → α) (t : TensorInputs) (r : Request)
    (m : Nat → Byte) (schedule : Schedule t r m) (scratch : FP.Structural.State α)
    (j : Job) : FP.Structural.State α :=
  { scratch with
    mem := WholeBacking.memory decode t r m schedule scratch
    pids := fun axisId => if axisId = 0 then schedule.index j else scratch.pids axisId
    numPids := fun axisId => if axisId = 0 then (jobs r).length else scratch.numPids axisId }

end SourceTensor

namespace SourceBacking

/-- Materialize exactly the observed source stores in one caller heap, using
the existing row order and byte layout. Readback and frame are proved in
`Proofs/WholeBacking.lean`; this definition assumes neither. -/
def memory {α : Type} [Inhabited α] (r : Request) (l : Layout r)
    (states : Job → FP.Structural.State α) (encode : α → Nat → Byte)
    (m : Nat → Byte) : Nat → Byte :=
  fun p => runStores ((stores r states).flatMap (expandStore r l encode))
    (fun a => m a.row) (physical p)

end SourceBacking
end
end VeriTile.Bench.Repairs.KvCacheFilling
