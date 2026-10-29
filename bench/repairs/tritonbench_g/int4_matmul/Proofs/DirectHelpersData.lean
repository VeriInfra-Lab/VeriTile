import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectMatmulNative
import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperReal

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Helpers HelperHost HelperBytes Physical
noncomputable section
set_option maxHeartbeats 2000000

structure QuantRequest where
  weight : TensorInfo
  group : Int
  rank : Int
  contract : QuantMetadataContract weight group rank
def QuantRequest.shape (r : QuantRequest) := quantShape r.weight r.group
def QuantRequest.valid (r : QuantRequest) : r.shape.Valid:=quantShape_valid r.weight r.group r.rank r.contract
structure UnpackRequest where
  weight : TensorInfo
  scales : TensorInfo
  zeros : TensorInfo
  contract : PackedContract weight scales zeros
def UnpackRequest.shape (r : UnpackRequest) := packedShape r.weight r.scales
def UnpackRequest.valid (r : UnpackRequest) : r.shape.Valid:=packedShape_valid r.weight r.scales r.zeros r.contract
def UnpackRequest.views (r : UnpackRequest) := [r.weight,r.scales,r.zeros]

def quantState (r : QuantRequest) (m : HeapState α) : State α :=
  ⟨fun _=>m.values r.weight.descriptor.allocation,fun _=>m.packed r.weight.descriptor.allocation⟩
def originalQuantState (r : QuantRequest) (m : HeapState α) : State α :=
  ⟨fun _=>logicalBuffer r.weight m.values,fun _=>m.packed r.weight.descriptor.allocation⟩
theorem quantState_original (r : QuantRequest) (m : HeapState α) : quantState r m=originalQuantState r m := by
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  refine ⟨?_,rfl⟩
  funext region i
  exact (logicalBuffer_eq r.weight .fp32 (Or.inl (canonical_tensor _ _ r.contract.weight)) m.values i).symm

def unpackState (r : UnpackRequest) (m : HeapState α) : State α :=
  ⟨fun _=>m.values r.scales.descriptor.allocation,fun region=>match region with
    | .weight=>m.packed r.weight.descriptor.allocation
    | .zero=>m.packed r.zeros.descriptor.allocation⟩
def originalUnpackState (r : UnpackRequest) (m : HeapState α) : State α :=
  ⟨fun _=>logicalBuffer r.scales m.values,fun region=>match region with
    | .weight=>logicalBuffer r.weight m.packed
    | .zero=>logicalBuffer r.zeros m.packed⟩
theorem unpackState_original (r : UnpackRequest) (m : HeapState α) : unpackState r m=originalUnpackState r m := by
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · funext region i
    exact (logicalBuffer_eq _ _ (packed_tensor _ _ r.contract.scales) m.values i).symm
  · funext region i
    cases region <;> simp only [unpackState,originalUnpackState,
      logicalBuffer_eq _ _ (packed_tensor _ _ r.contract.weight),logicalBuffer_eq _ _ (packed_tensor _ _ r.contract.zeros)]

def quantBytesInput (r : QuantRequest) (m : ByteHeap) : Bytes := fun _=>m r.weight.descriptor.allocation
def unpackBytesInput (r : UnpackRequest) (m : ByteHeap) : Bytes :=
  fun region=>match region with
  | .value _=>m r.scales.descriptor.allocation
  | .packed .weight=>m r.weight.descriptor.allocation
  | .packed .zero=>m r.zeros.descriptor.allocation
theorem quantBytesInput_load (r : QuantRequest) (m : ByteHeap) : load (quantBytesInput r m)=quantState r (heapWords m) := rfl
theorem unpackBytesInput_load (r : UnpackRequest) (m : ByteHeap) : load (unpackBytesInput r m)=unpackState r (heapWords m) := by
  change State.mk _ _=State.mk _ _
  rw [State.mk.injEq]
  constructor
  · rfl
  · funext region i;cases region <;> rfl

def QuantTensor (t : TensorInfo) (d : DType) (rows cols device : Nat) : Prop :=
  t.descriptor.TransposeContiguous d rows cols device ∧ t.strided=Bool.true ∧ t.cuda=Bool.true ∧ t.rank=2 ∧
    t.dataPtr.toNat%elementBytes d=0
structure QuantAllocation (r : QuantRequest) (live : Nat → Prop) where
  weights : TensorInfo
  scales : TensorInfo
  zeros : TensorInfo
  weightsDescriptor : QuantTensor weights .int32 (r.shape.k/8) r.shape.n r.weight.descriptor.device
  scalesDescriptor : QuantTensor scales .fp32 (r.shape.k/r.shape.group) r.shape.n r.weight.descriptor.device
  zerosDescriptor : QuantTensor zeros .int32 (r.shape.k/r.shape.group) (r.shape.n/8) r.weight.descriptor.device
  fresh : ∀ t,t∈[weights,scales,zeros] → ¬live t.descriptor.allocation
  separate : weights.descriptor.allocation≠scales.descriptor.allocation ∧ weights.descriptor.allocation≠zeros.descriptor.allocation ∧ scales.descriptor.allocation≠zeros.descriptor.allocation

theorem quantTensor_index (t : TensorInfo) (d : DType) (rows cols device row col : Nat)
    (h : QuantTensor t d rows cols device) (hr : row<rows) (hc : col<cols) : logicalIndex t row col=row*cols+col := by
  have offset:=h.1.2.2.2.2.2.2.1
  simp only [logicalIndex,offset,Nat.zero_add,t.descriptor.packed_active_offset d rows cols device row col (Or.inr h.1) hr hc]

def UnpackTensor (r : UnpackRequest) (t : TensorInfo) : Prop :=
  t.descriptor.rows=r.shape.k ∧ t.descriptor.columns=r.shape.n ∧ t.descriptor.rowStride=1 ∧ t.descriptor.columnStride=r.shape.k ∧
  t.descriptor.dtype=.fp16 ∧ t.descriptor.device=r.weight.descriptor.device ∧ t.descriptor.offset=0 ∧
  t.descriptor.lazyNeg=Bool.false ∧ t.descriptor.lazyConj=Bool.false ∧ t.descriptor.requiresGrad=Bool.false ∧
  t.strided=Bool.true ∧ t.cuda=Bool.true ∧ t.rank=2 ∧ t.dataPtr.toNat%2=0
structure UnpackAllocation (r : UnpackRequest) (live : Nat → Prop) where
  tensor : TensorInfo
  descriptor : UnpackTensor r tensor
  fresh : ¬live tensor.descriptor.allocation

theorem unpackTensor_index (r : UnpackRequest) (t : TensorInfo) (h : UnpackTensor r t) (k n : Nat) : logicalIndex t k n=n*r.shape.k+k := by
  simp only [logicalIndex,Tensor.elementOffset,h.2.2.1,h.2.2.2.1,h.2.2.2.2.2.2.1,Nat.zero_add,Nat.mul_one,Nat.add_comm,Nat.mul_comm]

structure HelperSource (r : QuantRequest) where
  order : ReductionOrder r.shape.group
  parsedProduction : String
  sourceIdentity : parsedProduction=SourceBinding.productionAST

#axiomsClean quantState_original
#axiomsClean unpackState_original
#axiomsClean quantBytesInput_load
#axiomsClean unpackBytesInput_load
#axiomsClean quantTensor_index
#axiomsClean unpackTensor_index
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
