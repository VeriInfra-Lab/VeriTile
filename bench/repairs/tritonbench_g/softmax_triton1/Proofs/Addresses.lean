import bench.repairs.tritonbench_g.softmax_triton1.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
open VeriTile.Triton
noncomputable section

/-- Physical buffer and canonical row descriptor retain the Tensor's actual
storage offset. No temporary input allocation is introduced. -/
def buffer (x : Object) : Native.Buffer :=
  ⟨x.tensor.storage.base,4*x.tensor.storage.elements,x.metadata.device.index⟩

/-- The formal pointer starts at the same data_ptr observed by admission. -/
theorem pointer_argument (x : Object) (positive : 0<x.metadata.numel) :
    BitVec.ofNat 64 (buffer x).base + BitVec.ofNat 64 (descriptor x).offset * BitVec.ofNat 64 4=
      x.tensor.dataPtr := by
  rw [x.laws.pointer positive]
  simp [buffer,descriptor,VeriTile.Torch.FP32.Storage.address,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,Nat.mul_comm]

theorem pointer_bounds (x : Object) (h : InputContract x.metadata)
    (r c : Nat) (hr : r<rows x.metadata) (hc : c<cols x.metadata) :
    (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat=
      x.tensor.storage.address (x.tensor.physicalIndex (r*cols x.metadata+c)) ∧
    (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat+4 ≤
      x.tensor.storage.base+4*x.tensor.storage.elements ∧
    (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat%4=0 := by
  have hi : r*cols x.metadata+c<x.metadata.numel := by
    rw [numel_eq _ h]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  have positive : 0<x.metadata.numel := by omega
  have hcB := hc.trans_le (blockSize_bounds _).2
  have abi := launch_bounds _ h positive r c hr hcB
  have physical := x.laws.contiguous h.contiguous _ hi
  have bound := x.laws.inBounds _ hi
  have span := x.tensor.storage.spanFits
  rw [physical] at bound
  have ptrBound : x.tensor.storage.base+4*(x.tensor.storageOffset+r*cols x.metadata+c)<2^64 := by
    omega
  have exactPtr : (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat=
      x.tensor.storage.base+4*(x.tensor.storageOffset+r*cols x.metadata+c) := by
    rw [Native.source_sequence_same]
    simp only [Native.sourcePointer,Native.index64_exact r abi.1,Native.index64_exact c abi.2.1,
      buffer,descriptor,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat,
      Nat.mul_one]
    rw [Nat.mul_comm (x.tensor.storageOffset+r*cols x.metadata+c) 4,Nat.mod_eq_of_lt ptrBound]
  have inputPointer := x.laws.pointer positive
  have firstBound : x.tensor.storage.address x.tensor.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have aligned := h.alignment positive
  change x.tensor.dataPtr.toNat%4=0 at aligned
  rw [inputPointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt firstBound] at aligned
  rw [exactPtr]
  refine ⟨?_,by omega,?_⟩
  · rw [physical]
    simp [VeriTile.Torch.FP32.Storage.address,Nat.add_assoc]
  · simp only [VeriTile.Torch.FP32.Storage.address] at aligned
    omega

theorem mask_exact (x : Object) (h : InputContract x.metadata) (positive : 0<x.metadata.numel)
    (r c : Nat) (hr : r<rows x.metadata) (hc : c<blockSize (cols x.metadata)) :
    (Native.index64 c).slt (BitVec.ofNat 64 (cols x.metadata))=decide (c<cols x.metadata) := by
  have abi := launch_bounds _ h positive r c hr hc
  exact Native.source_mask_exact _ _ abi.2.1 abi.2.2

#axiomsClean pointer_argument
#axiomsClean pointer_bounds
#axiomsClean mask_exact
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
