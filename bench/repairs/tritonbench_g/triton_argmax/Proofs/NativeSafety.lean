import bench.repairs.tritonbench_g.triton_argmax.Proofs.TensorBinding
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Native

namespace VeriTile.Bench.Repairs.TritonArgmax.NativeSafety
open Source Memory TensorBinding
noncomputable section

/-- The native uint64 pointer is the actual rebased Tensor pointer; each
accessed word is aligned and its complete byte span remains in the allocation. -/
def InputAccess (x : Input) : Prop := ∀ i, i < x.tensor.metadata.numel →
  (BitVec.ofNat 64 (x.tensor.metadata.dataPtr.toNat+4*i)).toNat =
    (address x.host.dataWindow 4 i).2 ∧
  (address x.host.dataWindow 4 i).2+4 ≤ x.storage.base+x.storage.bytes ∧
  (address x.host.dataWindow 4 i).2+4 ≤ 2^64 ∧
  (address x.host.dataWindow 4 i).2 % 4 = 0

theorem input_access (x : Input) (contiguous : x.tensor.metadata.isContiguous = Bool.true)
    (checked : 0 < x.tensor.metadata.numel → x.tensor.metadata.dataPtr.toNat % 4 = 0) :
    InputAccess x := by
  intro i hi
  have positive : 0 < x.tensor.metadata.numel := by omega
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor x.laws contiguous positive
  have span := x.tensor.storage.spanFits
  have pointer := x.laws.pointer positive
  have baseBound : x.tensor.storage.base+4*x.tensor.storageOffset < 2^64 := by omega
  have ptr : x.tensor.metadata.dataPtr.toNat = x.tensor.storage.base+4*x.tensor.storageOffset := by
    simpa only [VeriTile.Torch.FP32.Tensor.dataPtr,VeriTile.Torch.FP32.Storage.address,
      BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound] using congrArg BitVec.toNat pointer
  have aligned := checked positive
  rw [ptr] at aligned
  change (BitVec.ofNat 64 (x.tensor.metadata.dataPtr.toNat+4*i)).toNat =
      x.tensor.storage.base+4*x.tensor.storageOffset+4*i ∧
    x.tensor.storage.base+4*x.tensor.storageOffset+4*i+4 ≤ x.tensor.storage.base+4*x.tensor.storage.elements ∧
    x.tensor.storage.base+4*x.tensor.storageOffset+4*i+4 ≤ 2^64 ∧
    (x.tensor.storage.base+4*x.tensor.storageOffset+4*i)%4 = 0
  rw [ptr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega :
    x.tensor.storage.base+4*x.tensor.storageOffset+4*i < 2^64)]
  exact ⟨rfl,by omega,by omega,by simpa [Nat.add_mod] using aligned⟩

def AllocatedAccess (x : AllocatedTensor) : Prop := ∀ i, i < x.metadata.numel →
  (BitVec.ofNat 64 (x.metadata.dataPtr.toNat+x.width*i)).toNat =
    (address x.storage x.width i).2 ∧
  (address x.storage x.width i).2+x.width ≤ x.storage.base+x.storage.bytes ∧
  (address x.storage x.width i).2+x.width ≤ 2^64 ∧
  (address x.storage x.width i).2 % min x.width 8 = 0

theorem allocated_access (x : AllocatedTensor) (positiveWidth : 0 < x.width)
    (widthBound : x.width ≤ 8) : AllocatedAccess x := by
  intro i hi
  have positive : 0 < x.metadata.numel := by omega
  have next := Nat.mul_le_mul_left x.width (show i+1 ≤ x.metadata.numel by omega)
  have capacity := x.capacity
  have span := x.span
  have baseBound : x.storage.base < 2^64 := by nlinarith
  have ptr : x.metadata.dataPtr.toNat = x.storage.base := by
    simpa only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound] using
      congrArg BitVec.toNat (x.pointer positive)
  have aligned := x.checkedAlignment positive
  rw [ptr,Nat.min_eq_left widthBound] at aligned
  change (BitVec.ofNat 64 (x.metadata.dataPtr.toNat+x.width*i)).toNat =
      x.storage.base+x.width*i ∧ _
  rw [ptr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by nlinarith : x.storage.base+x.width*i < 2^64)]
  refine ⟨rfl,by dsimp [address]; nlinarith,by dsimp [address]; nlinarith,?_⟩
  simpa [address,Nat.min_eq_left widthBound,Nat.add_mod] using aligned

theorem format_width_bound (arg : Checked.DTypeArgument) (a : RealScratch.Adapter)
    (dtype : arg.adapter = some a) : 0 < a.format.bytes ∧ a.format.bytes ≤ 8 := by
  cases arg with
  | omitted => cases dtype; decide
  | floating kind => cases dtype; cases kind <;> decide
  | integer kind => cases dtype; cases kind <;> decide
  | invalid token => cases dtype

structure Global (x : Input) (o : Checked.Observation) (a : GlobalAllocation x o d callers) : Prop where
  input : InputAccess x
  midValue : AllocatedAccess a.midValue
  midIndex : AllocatedAccess a.midIndex
  output : AllocatedAccess a.output.returned
  index : ∀ pid lane, pid < Checked.blocks o → lane < Checked.block o →
    Native.signed (lane+pid*Checked.block o) = Int.ofNat (lane+pid*Checked.block o)
  gather : ∀ lane : Fin (Checked.midBlock o), lane.val < a.midIndex.metadata.numel

theorem global (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.GlobalContract o) (adapter : RealScratch.Adapter)
    (adapterDtype : o.dtype.adapter = some adapter)
    (a : GlobalAllocation x o adapter.format callers) : Global x o a := by
  rcases h with ⟨contiguous,noNeg,nonempty,dtype,blockPos,blockMax,blocksPos,blocksFit,midMax,
    padded,indexBytes,aligned,pointerSpan⟩
  refine ⟨input_access x (observed ▸ contiguous) (fun _ => observed ▸ aligned),
    allocated_access a.midValue ?_ ?_,allocated_access a.midIndex ?_ ?_,
    allocated_access a.output.returned ?_ ?_,?_,?_⟩
  · rw [a.midWidth]; exact (format_width_bound o.dtype adapter adapterDtype).1
  · rw [a.midWidth]; exact (format_width_bound o.dtype adapter adapterDtype).2
  · rw [a.indexWidth]; omega
  · rw [a.indexWidth]
  · rw [a.output.width]; omega
  · rw [a.output.width]
  · intro pid lane hp hl
    have mul := Nat.mul_lt_mul_of_pos_right hp blockPos
    have tail := Nat.div_mul_le_self (o.input.numel+Checked.block o-1) (Checked.block o)
    apply Native.signed_exact
    change lane+pid*Checked.block o < 2^63
    change pid*Checked.block o < Checked.blocks o*Checked.block o at mul
    dsimp [Checked.blocks] at mul
    omega
  · intro lane
    simpa [VeriTile.Torch.TensorMetadata.numel,a.indexShape] using lane.isLt

structure Dimension (o : Checked.Observation) (ax : Nat) (prepared : Input)
    (out : AllocatedTensor) : Prop where
  input : 0 < Checked.outer o ax ∧ 0 < Checked.inner o ax → InputAccess prepared
  output : AllocatedAccess out
  masks : ∀ row col, 0 < Checked.outer o ax ∧ 0 < Checked.inner o ax →
    row < Checked.outer o ax+128 → col < Checked.extent o ax+128 →
    ((Native.signed row < Native.signed (Checked.outer o ax) ∧
      Native.signed col < Native.signed (Checked.extent o ax)) ↔
      row < Checked.outer o ax ∧ col < Checked.extent o ax)
  inputArithmetic : ∀ row col k, row < Checked.outer o ax → col < Checked.extent o ax →
    k < Checked.inner o ax → Native.dimOffset row col k (Checked.extent o ax) (Checked.inner o ax) =
      row*Checked.extent o ax*Checked.inner o ax+col*Checked.inner o ax+k
  outputArithmetic : ∀ row k, row < Checked.outer o ax → k < Checked.inner o ax →
    Native.signed (row*Checked.inner o ax+k) = Int.ofNat (row*Checked.inner o ax+k)

theorem dimension (x : Input) (o : Checked.Observation) (observed : o.input = x.tensor.metadata)
    (h : Checked.BaseContract o) (ax : Nat) (valid : ax < x.tensor.metadata.shape.length)
    (hd : Checked.DimensionContract o ax) (call : ContiguousCall mem before x callers)
    (prepared : o.prepared = call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) :
    Dimension o ax call.result a.output.returned := by
  have factor : Checked.outer o ax*Checked.extent o ax*Checked.inner o ax = o.input.numel := by
    simpa only [Checked.outer,Checked.extent,Checked.inner,VeriTile.Torch.TensorMetadata.numel,observed]
      using RequestBinding.shape_factorization x.tensor.metadata.shape ax valid
  refine ⟨?_,allocated_access a.output.returned (by rw [a.output.width]; omega)
    (by rw [a.output.width]),?_,?_,?_⟩
  · intro active
    apply input_access call.result call.contiguous
    intro _
    exact prepared ▸ (hd.2 active).2.2.2.2.2.2.1
  · intro row col active hr hc
    have bounds := hd.2 active
    apply Native.exact_mask <;> change _ < 2^63 <;> omega
  · intro row col k hr hc hk
    have bound : 4*(Checked.outer o ax*Checked.extent o ax*Checked.inner o ax) < Native.signedLimit := by
      rw [factor]
      exact h.2.2.2
    exact (Native.active_dim_address _ _ _ row col k hr hc hk bound).1
  · intro row k hr hk
    exact Native.active_output_address _ _ row k hr hk (hd.2 ⟨by omega,by omega⟩).1

theorem empty_dimension (g : NativeGraph.DimensionPrograms) (o : Checked.Observation) (ax : Nat)
    (mem : Memory.Primitives M) (buffers : DimMemory.Buffers) (state : M)
    (empty : Checked.outer o ax = 0 ∨ Checked.inner o ax = 0) :
    NativeGraph.dimension g o ax mem buffers state = state := by
  rcases empty with outer | inner
  · simp [NativeGraph.dimension,SegmentedExecution.dimJobs,NativeGraph.DimensionPrograms.config,
      outer,SegmentedExecution.schedule]
  · simp [NativeGraph.dimension,SegmentedExecution.dimJobs,NativeGraph.DimensionPrograms.config,
      inner,SegmentedExecution.schedule]

#axiomsClean input_access
#axiomsClean allocated_access
#axiomsClean global
#axiomsClean dimension
#axiomsClean empty_dimension
end
end VeriTile.Bench.Repairs.TritonArgmax.NativeSafety
