import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Contract
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Storage

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar
open VeriTile.Triton VeriTile.Torch
open scoped BigOperators

theorem contract_iff (s : TensorMetadata) (dtype : Option DType) :
    InputContract s dtype ↔ s.layout = .strided ∧ s.dtype = .floating .fp32 ∧
      (dtype = none ∨ dtype = some (.floating .fp32)) ∧ s.shape.length = 3 ∧
      s.device.kind = "cuda" ∧ s.isContiguous = Bool.true ∧ s.isNeg = Bool.false ∧
      (0 < s.numel → rows s < 2^31) ∧ (0 < s.numel → s.dataPtr.toNat % 4 = 0) := by
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

/-- Bind the source arguments to the original contiguous Tensor's metadata and
actual strides. These are Tensor/empty_like laws, with no execution or output
formula. Singleton dimensions may have noncanonical strides. -/
structure TensorBinding (s : TensorMetadata) (h : Call) (p : Storage.Placement h) : Prop where
  shape : s.shape = [h.B,h.H,h.T]
  inputDType : h.inputDType = .fp32
  outputDType : h.outputDType = .fp32
  inputPointer : 0 < s.numel → p.backing.input.base = s.dataPtr.toNat
  inputFlat : ∀ row time, row < h.B*h.H → time < h.T →
    h.inputStrides.address h.H row time = row*h.T+time
  outputFlat : ∀ row time, row < h.B*h.H → time < h.T →
    h.outputStrides.address h.H row time = row*h.T+time
  distinct : h.input ≠ h.output
  outputAlignment : 0 < s.numel → p.backing.output.base % 4 = 0

namespace TensorBinding
theorem outputLayout (s : TensorMetadata) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) : Geometry.OutputLayout h := by
  refine ⟨binding.distinct,?_⟩
  intro r q t u hr hq ht hu eq
  rw [binding.outputFlat r t hr ht,binding.outputFlat q u hq hu] at eq
  have same : r=q := by
    rcases lt_trichotomy r q with less | same | greater
    · have gap := Nat.mul_le_mul_right h.T (Nat.succ_le_of_lt less)
      rw [Nat.succ_mul] at gap
      omega
    · exact same
    · have gap := Nat.mul_le_mul_right h.T (Nat.succ_le_of_lt greater)
      rw [Nat.succ_mul] at gap
      omega
  subst q
  exact ⟨rfl,by omega⟩

theorem numel (s : TensorMetadata) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) : s.numel=h.B*h.H*h.T := by
  simp [TensorMetadata.numel,binding.shape,Nat.mul_assoc]

def pid (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<h.B*h.H*h.T) (row : Fin (h.B*h.H)) : NativeIndex.PID h := by
  have bound := contract.grid (by rw [binding.numel]; exact nonempty)
  have rowCount : rows s=h.B*h.H := by simp [rows,binding.shape]
  exact ⟨row.val,row.isLt,by rw [rowCount] at bound; exact row.isLt.trans bound⟩

theorem aligned (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<s.numel) (offset : Nat) :
    Storage.address p.backing.input offset % 4 = 0 := by
  simp [Storage.address,p.inputBytes,binding.inputPointer nonempty,Nat.add_mod,contract.alignment nonempty]
end TensorBinding

#axiomsClean checkInput_iff
#axiomsClean TensorBinding.outputLayout
#axiomsClean TensorBinding.pid
#axiomsClean TensorBinding.aligned
end VeriTile.Bench.Repairs.ReversedCumsumScalar
