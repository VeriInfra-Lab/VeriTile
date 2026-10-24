import bench.repairs.tritonbench_g.matmul_persistent_triton.Contract
import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBinding
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Torch
noncomputable section

/-- Actual Torch object observations, including the API's independent logical
index map. These fields contain no kernel, arithmetic, result, or execution. -/
structure ActualTensor where
  object : Nat
  metadata : TensorMetadata
  storage : Nat
  storageOffset : Nat
  rowStride : Nat
  columnStride : Nat
  isConj : Bool
  physicalIndex : Nat → Nat

def torchType : Storage → DType
  | .fp16 => .other "torch.float16"
  | .f8e4 => .other "torch.float8_e4m3fn"

/-- Ordinary strided API binding; a kernel output value is never a premise. -/
structure Laws (descriptor : Tensor) (actual : ActualTensor) : Prop where
  object : actual.object=descriptor.object
  dtype : actual.metadata.dtype=torchType descriptor.dtype
  layout : actual.metadata.layout=.strided
  deviceKind : actual.metadata.device.kind="cuda"
  deviceIndex : actual.metadata.device.index=descriptor.device
  shape : actual.metadata.shape=[descriptor.rows,descriptor.columns]
  storage : actual.storage=descriptor.storage
  offset : actual.storageOffset=descriptor.offset
  rowStride : actual.rowStride=descriptor.rowStride
  columnStride : actual.columnStride=descriptor.columnStride
  pointer : actual.metadata.dataPtr=descriptor.pointer
  isNeg : actual.metadata.isNeg=descriptor.isNeg
  isConj : actual.isConj=descriptor.isConj
  indexing : ∀ row<descriptor.rows,∀ column<descriptor.columns,
    actual.physicalIndex (row*descriptor.columns+column)=
      descriptor.offset+row*descriptor.rowStride+column*descriptor.columnStride

def logicalValue {α : Type} (memory : Memory α) (actual : ActualTensor) (index : Nat) : α :=
  memory (actual.storage,actual.physicalIndex index)

theorem logical_read {α : Type} (memory : Memory α) (descriptor : Tensor)
    (actual : ActualTensor) (laws : Laws descriptor actual)
    (row column : Nat) (hr : row<descriptor.rows) (hc : column<descriptor.columns) :
    logicalValue memory actual (row*descriptor.columns+column)=memory (descriptor.address row column) := by
  simp only [logicalValue,laws.storage,laws.indexing row hr column hc,Tensor.address]

structure Arguments where
  descriptors : Buffers
  a : ActualTensor
  b : ActualTensor
  output : ActualTensor
  aLaws : Laws descriptors.A a
  bLaws : Laws descriptors.B b
  outputLaws : Laws descriptors.C output

/-- One raw element observation for every physical storage index. It retains
input aliasing and the entire output-storage exterior, not only live cells. -/
def PhysicalFrame {α : Type} (p : Problem) (b : Buffers) (before after : Memory α) : Prop :=
  ∀ point,(∀ r<p.M,∀ c<p.N,b.C.address r c≠point) → after point=before point

theorem input_storage_frame {α : Type} (p : Problem) (b : Buffers)
    (before after : Memory α) (frame : PhysicalFrame p b before after)
    (separateA : b.C.storage≠b.A.storage) (separateB : b.C.storage≠b.B.storage) :
    (∀ i,after (b.A.storage,i)=before (b.A.storage,i)) ∧
    (∀ i,after (b.B.storage,i)=before (b.B.storage,i)) := by
  constructor <;> intro i <;> apply frame
  · intro r hr c hc same
    exact separateA (congrArg Prod.fst same)
  · intro r hr c hc same
    exact separateB (congrArg Prod.fst same)

#axiomsClean logical_read
#axiomsClean input_storage_frame
#specNonCircular Laws avoiding [Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.TensorBinding
