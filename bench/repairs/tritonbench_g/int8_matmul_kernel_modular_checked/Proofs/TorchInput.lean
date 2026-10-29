import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

/-- Ordinary binding of the existing rank-two descriptor to the shared FP32
Tensor API. The element map is the standard strided storage-offset equation;
none of these fields mentions kernel execution or an arithmetic result. -/
structure FP32Binding (info : Tensor) (pointer : Nat) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  storage : tensor.storage.object = info.storage
  offset : tensor.storageOffset = info.offset
  extent : tensor.storage.elements = info.capacity
  shape : tensor.metadata.shape = [info.rows,info.cols]
  resolved : tensor.metadata.isNeg = !info.resolved
  dataPointer : tensor.dataPtr.toNat = pointer
  strided : ∀ m k, m < info.rows → k < info.cols →
    tensor.physicalIndex (m*info.cols+k) = info.offset+m*info.strideRow+k*info.strideCol

/-- Read FP32 through the same typed common backing, using the shared
Tensor storage's absolute byte address. Integer Tensor carriers remain the
existing task-local descriptors because no shared UINT8/INT32 API exists. -/
def fp32Read (memory : Memory Word32) (tensor : VeriTile.Torch.FP32.Tensor)
    (address : Nat) : VeriTile.Torch.FP32.Word :=
  ⟨memory ((tensor.storage.object,.fp32),(address-tensor.storage.base)/4)⟩

theorem fp32_at_index (memory : Memory Word32) (tensor : VeriTile.Torch.FP32.Tensor) (i : Nat) :
    (fp32Read memory tensor (tensor.storage.address i)).bits =
      memory ((tensor.storage.object,.fp32),i) := by
  simp [fp32Read,VeriTile.Torch.FP32.Storage.address]

/-- The source data pointer plus its actual row/inner strides is the shared
Tensor's physical FP32 address, including a nonzero storage offset. -/
theorem torch_source_address (info : Tensor) (pointer : Nat)
    (binding : FP32Binding info pointer) (m k : Nat) (hm : m < info.rows) (hk : k < info.cols) :
    binding.tensor.storage.address (binding.tensor.physicalIndex (m*info.cols+k)) =
      pointer+4*(m*info.strideRow+k*info.strideCol) := by
  have numel : binding.tensor.metadata.numel = info.rows*info.cols := by
    simp [VeriTile.Torch.TensorMetadata.numel,binding.shape]
  have hi : m*info.cols+k < binding.tensor.metadata.numel := by
    rw [numel]
    have hmul := Nat.mul_le_mul_right info.cols (Nat.succ_le_of_lt hm)
    nlinarith
  have cap := binding.laws.inBounds _ hi
  rw [binding.strided m k hm hk] at cap
  have span := binding.tensor.storage.spanFits
  have bound : binding.tensor.storage.address binding.tensor.storageOffset < 2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address,binding.offset]
    omega
  have pointerEq := congrArg BitVec.toNat (binding.laws.pointer (Nat.zero_lt_of_lt hi))
  rw [binding.dataPointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at pointerEq
  rw [binding.strided m k hm hk]
  calc
    _ = binding.tensor.storage.address binding.tensor.storageOffset +
        4*(m*info.strideRow+k*info.strideCol) := by
      simp [VeriTile.Torch.FP32.Storage.address,binding.offset,Nat.mul_add,Nat.add_assoc]
    _ = pointer+4*(m*info.strideRow+k*info.strideCol) :=
      congrArg (fun base => base+4*(m*info.strideRow+k*info.strideCol)) pointerEq.symm

theorem torch_input_word (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (binding : FP32Binding request.a aPointer)
    (admitted : InputContract request aPointer bPointer before)
    (m k : Nat) (hm : m < request.shape.m) (hk : k < request.shape.k) :
    (VeriTile.Torch.FP32.logicalValue (fp32Read before binding.tensor)
      binding.tensor binding.laws (m*request.a.cols+k)).bits =
      before (aAddress (hostBuffers request out) m k) := by
  have resolved : request.a.resolved = Bool.true := admitted.metadata.2.2.2.2.2.2.2.2.1
  have notNeg : binding.tensor.metadata.isNeg = Bool.false := by
    rw [binding.resolved,resolved]
    rfl
  simp only [VeriTile.Torch.FP32.logicalValue,notNeg,Bool.false_eq_true,ite_false]
  rw [fp32_at_index,binding.strided m k hm hk,binding.storage]
  rfl

theorem source_torch_input (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (binding : FP32Binding request.a aPointer)
    (admitted : InputContract request aPointer bPointer before)
    (m k : Nat) (hm : m < request.shape.m) (hk : k < request.shape.k) :
    inputWords (layout request) (encodedState request out before) m k =
      (VeriTile.Torch.FP32.logicalValue (fp32Read before binding.tensor)
        binding.tensor binding.laws (m*request.a.cols+k)).bits := by
  rw [input_encoding,torch_input_word request aPointer bPointer out before binding admitted m k hm hk]

def torchSignedSum (request : Request) (aPointer out : Nat) (before : Memory Word32)
    (binding : FP32Binding request.a aPointer) (m n : Nat) : Int :=
  denseSignedSum request.shape.k
    (fun k => (VeriTile.Torch.FP32.logicalValue (fp32Read before binding.tensor)
      binding.tensor binding.laws (m*request.a.cols+k)).bits)
    (fun q => before (bAddress (hostBuffers request out) q n))

theorem torch_target (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (binding : FP32Binding request.a aPointer)
    (admitted : InputContract request aPointer bPointer before)
    (m n : Nat) (hm : m < request.shape.m) :
    torchSignedSum request aPointer out before binding m n =
      tensorSignedSum request out before m n := by
  unfold torchSignedSum tensorSignedSum denseSignedSum
  apply Finset.sum_congr rfl
  intro f hf
  apply Finset.sum_congr rfl
  intro q hq
  have hf' : f < 4 := Finset.mem_range.mp hf
  have hq' : q < request.shape.k/4 := Finset.mem_range.mp hq
  have hk : f*(request.shape.k/4)+q < request.shape.k := by
    have hdiv := Nat.mod_add_div request.shape.k 4
    have hmul := Nat.mul_le_mul_right (request.shape.k/4) (show f ≤ 3 by omega)
    omega
  dsimp only
  rw [torch_input_word request aPointer bPointer out before binding admitted m _ hm hk]

#axiomsClean fp32_at_index
#axiomsClean torch_source_address
#axiomsClean torch_input_word
#axiomsClean source_torch_input
#axiomsClean torch_target
#specNonCircular torchSignedSum avoiding [encodedState, resultMemory, source, gridMemory, modularAccumulation]
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
