import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Contract
import VeriTile.Torch.FP32

/-! Tensor storage laws bind independent logical coordinates to the exact
allocation-relative offsets used by both source passes. These are input/API
facts; no execution or numerical result occurs in the binding. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section

abbrev ActualTensor := VeriTile.Torch.FP32.Tensor

def flatIndex (p : Parameters) (b m k : Nat) : Nat := (b*p.rows+m)*p.width+k

def tensorRead (memory : Memory α) (t : ActualTensor) (i : Nat) : α :=
  memory t.storage.object (t.physicalIndex i)

/-- Actual Tensor objects, their API metadata and logical-to-storage mappings.
The stride laws are precisely the standard strided indexing rule. -/
structure TensorBinding (x w out : TensorInfo) (p : Parameters) (byteBase : Nat → Nat) where
  input : ActualTensor
  weight : ActualTensor
  output : ActualTensor
  inputLaws : VeriTile.Torch.FP32.TensorLaws input
  weightLaws : VeriTile.Torch.FP32.TensorLaws weight
  outputLaws : VeriTile.Torch.FP32.TensorLaws output
  inputMetadata : input.metadata=x.toTensorMetadata
  weightMetadata : weight.metadata=w.toTensorMetadata
  outputMetadata : output.metadata=out.toTensorMetadata
  inputAllocation : input.storage.object=p.input.allocation
  weightAllocation : weight.storage.object=p.weights.allocation
  outputAllocation : output.storage.object=p.output.allocation
  inputBase : input.storage.base=byteBase p.input.allocation
  weightBase : weight.storage.base=byteBase p.weights.allocation
  outputBase : output.storage.base=byteBase p.output.allocation
  inputIndex : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    input.physicalIndex (flatIndex p b m k)=inputOffset p b m k
  weightIndex : ∀ k,k<p.width → weight.physicalIndex k=weightOffset p k
  outputIndex : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
    output.physicalIndex (flatIndex p b m k)=outputOffset p b m k

namespace TensorBinding

theorem input_read (v : TensorBinding x w out p base) (memory : Memory α)
    (b m k : Nat) (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    tensorRead memory v.input (flatIndex p b m k)=memory p.input.allocation (inputOffset p b m k) := by
  simp only [tensorRead,v.inputAllocation,v.inputIndex b m k hb hm hk]

theorem weight_read (v : TensorBinding x w out p base) (memory : Memory α)
    (k : Nat) (hk : k<p.width) :
    tensorRead memory v.weight k=memory p.weights.allocation (weightOffset p k) := by
  simp only [tensorRead,v.weightAllocation,v.weightIndex k hk]

theorem output_read (v : TensorBinding x w out p base) (memory : Memory α)
    (b m k : Nat) (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    tensorRead memory v.output (flatIndex p b m k)=memory p.output.allocation (outputOffset p b m k) := by
  simp only [tensorRead,v.outputAllocation,v.outputIndex b m k hb hm hk]

/-- Exact source pointer expression, including the original storage offset.
No contiguous-copy or reset-to-zero storage offset is introduced. -/
theorem input_pointer (v : TensorBinding x w out p base)
    (binding : Binding x w out eps p base) (b m k : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    v.input.storage.address (v.input.physicalIndex (flatIndex p b m k)) =
      v.input.dataPtr.toNat+4*(b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2) := by
  rw [v.inputIndex b m k hb hm hk]
  simp only [VeriTile.Torch.FP32.Storage.address,v.inputBase,
    VeriTile.Torch.FP32.Tensor.dataPtr,v.inputMetadata]
  rw [←binding.input.pointer]
  simp only [inputOffset]
  omega

theorem weight_pointer (v : TensorBinding x w out p base)
    (binding : Binding x w out eps p base) (k : Nat) (hk : k<p.width) :
    v.weight.storage.address (v.weight.physicalIndex k) =
      v.weight.dataPtr.toNat+4*(k*p.weights.stride 0) := by
  rw [v.weightIndex k hk]
  simp only [VeriTile.Torch.FP32.Storage.address,v.weightBase,
    VeriTile.Torch.FP32.Tensor.dataPtr,v.weightMetadata]
  rw [←binding.weight.pointer]
  simp only [weightOffset]
  omega

theorem output_pointer (v : TensorBinding x w out p base)
    (binding : Binding x w out eps p base) (b m k : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (hk : k<p.width) :
    v.output.storage.address (v.output.physicalIndex (flatIndex p b m k)) =
      v.output.dataPtr.toNat+4*(b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2) := by
  rw [v.outputIndex b m k hb hm hk]
  simp only [VeriTile.Torch.FP32.Storage.address,v.outputBase,
    VeriTile.Torch.FP32.Tensor.dataPtr,v.outputMetadata]
  rw [←binding.output.pointer]
  simp only [outputOffset]
  omega
end TensorBinding

#axiomsClean TensorBinding.input_read
#axiomsClean TensorBinding.input_pointer
#axiomsClean TensorBinding.weight_pointer
#axiomsClean TensorBinding.output_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation
