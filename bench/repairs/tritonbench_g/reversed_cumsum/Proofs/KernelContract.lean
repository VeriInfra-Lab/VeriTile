import bench.repairs.tritonbench_g.reversed_cumsum.Contract
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsum
open VeriTile.Triton VeriTile.Torch
open scoped BigOperators

theorem contract_iff (s : TensorMetadata) (dtype : Option DType) :
    InputContract s dtype ↔ s.layout = .strided ∧ s.dtype = .floating .fp32 ∧
      (dtype = none ∨ dtype = some (.floating .fp32)) ∧ s.shape.length = 4 ∧
      s.device.kind = "cuda" ∧ s.isContiguous = Bool.true ∧ s.isNeg = Bool.false ∧
      (0 < s.numel → (rows s < 2^31 ∧ featureBlocks s < 2^31)) ∧ (0 < s.numel → s.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h
    exact ⟨h.layout,h.inputDType,h.outputDType,h.rank,h.cuda,h.contiguous,h.noNeg,h.grid,h.alignment⟩
  · rintro ⟨hl,hi,ho,hr,hc,hco,hn,hg,ha⟩
    exact ⟨hl,hi,ho,hr,hc,hco,hn,hg,ha⟩

theorem checkTensorInput_iff (s : TensorMetadata) (dtype : Option DType) :
    checkTensorInput s dtype = .ok () ↔ InputContract s dtype := by
  rw [contract_iff]
  simp [checkTensorInput, TensorMetadata.isCuda, ite_eq_iff]
  tauto

theorem checkInput_iff (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype = .ok () ↔ AdmittedArgument s dtype := by
  cases s <;> simp [checkInput,AdmittedArgument,checkTensorInput_iff]

/-- The modeled source arguments and backing pointer are those of the actual
original contiguous Tensor. Canonical active strides are passed explicitly. -/
structure TensorBinding (s : TensorMetadata) (h : Call) (p : Storage.Placement h) : Prop where
  shape : s.shape=[h.B,h.H,h.T,h.S]
  inputDType : h.inputDType=.fp32
  outputDType : h.outputDType=.fp32
  inputPointer : 0<s.numel → p.backing.input.base=s.dataPtr.toNat
  distinct : h.input≠h.output
  outputAlignment : 0<s.numel → p.backing.output.base%4=0

namespace TensorBinding
theorem numel (s : TensorMetadata) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) : s.numel=h.B*h.H*(h.T*h.S) := by
  simp [TensorMetadata.numel,binding.shape,Nat.mul_assoc]

def pid (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<h.B*h.H*(h.T*h.S)) (program : Geometry.Program h) : NativeIndex.PID h := by
  have bound := contract.grid (by rw [binding.numel]; exact nonempty)
  have rowCount : rows s=h.B*h.H := by simp [rows,binding.shape]
  have featureCount : featureBlocks s=Geometry.featureChunks h := by
    simp [featureBlocks,binding.shape,Geometry.featureChunks,ReversedCumsumScalar.Geometry.chunks,
      Call.featureView,Call.flat,BS]
  exact ⟨program.1,program.2,by rw [rowCount] at bound; exact program.1.isLt.trans bound.1,
    by rw [featureCount] at bound; exact program.2.isLt.trans bound.2⟩

theorem aligned (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<s.numel) (offset : Nat) :
    Storage.address p.backing.input offset%4=0 := by
  simp [Storage.address,ReversedCumsumScalar.Storage.address,p.inputBytes,binding.inputPointer nonempty,Nat.add_mod,contract.alignment nonempty]
end TensorBinding

#axiomsClean checkInput_iff
#axiomsClean TensorBinding.pid
#axiomsClean TensorBinding.aligned
end VeriTile.Bench.Repairs.ReversedCumsum
