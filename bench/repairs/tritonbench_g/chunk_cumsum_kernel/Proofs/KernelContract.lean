import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Storage
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Triton VeriTile.Torch

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

/-- Tensor dimensions and the actual admitted data pointer. Contiguous Tensor
coordinate readback is supplied locally by the public logical-input law. -/
structure TensorBinding (s : TensorMetadata) (c : Layout.Shape) (p : Storage.Placement c) : Prop where
  shape : s.shape=[c.B,c.H,c.T]
  inputPointer : 0<s.numel → p.backing.input.base=s.dataPtr.toNat
  outputAlignment : 0<s.numel → p.backing.output.base%4=0

/-- Ordinary contiguous Tensor input readback from the current physical heap.
This law states only input representation; it mentions no kernel or output.
The same law is used with Real cells and with typed FP payload cells. -/
structure InputReadLaw {β γ : Type} (s : TensorMetadata) (c : Layout.Shape)
    (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → γ) : Prop where
  reads : ∀ row time, row<c.B*c.H → time<c.T →
    read (heap (s.dataPtr.toNat+(row*c.T+time)*4))=logical row time

namespace TensorBinding
theorem input_read {β : Type} (s : TensorMetadata) (c : Layout.Shape)
    (p : Storage.Placement c) (binding : TensorBinding s c p)
    (nonempty : 0<s.numel) (other : β) (heap : Nat → β) (offset : Nat) :
    Storage.lift c p other heap p.input offset = heap (s.dataPtr.toNat+offset*4) := by
  rw [Storage.lift_input]
  simp only [Storage.address,binding.inputPointer nonempty]

theorem numel (s : TensorMetadata) (c : Layout.Shape) (p : Storage.Placement c)
    (binding : TensorBinding s c p) : s.numel=c.B*c.H*c.T := by
  simp [TensorMetadata.numel,binding.shape,Nat.mul_assoc]

theorem physical_input_read {β γ : Type} (s : TensorMetadata) (c : Layout.Shape)
    (p : Storage.Placement c) (binding : TensorBinding s c p)
    (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → γ)
    (law : InputReadLaw s c heap read logical)
    (row time : Nat) (hr : row<c.B*c.H) (ht : time<c.T) :
    read (heap (Storage.address p.backing.input (row*c.T+time)))=logical row time := by
  have nonempty : 0<s.numel := by
    rw [binding.numel]
    exact Nat.mul_pos (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt ht)
  simpa only [Storage.address,binding.inputPointer nonempty] using law.reads row time hr ht

theorem logical_input_read {β γ : Type} (s : TensorMetadata) (c : Layout.Shape)
    (p : Storage.Placement c) (binding : TensorBinding s c p)
    (other : β) (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → γ)
    (law : InputReadLaw s c heap read logical)
    (row time : Nat) (hr : row<c.B*c.H) (ht : time<c.T) :
    read (Storage.lift c p other heap p.input (row*c.T+time))=logical row time := by
  rw [Storage.lift_input]
  exact physical_input_read s c p binding heap read logical law row time hr ht

def pid (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (c : Layout.Shape) (p : Storage.Placement c) (binding : TensorBinding s c p)
    (nonempty : 0<c.B*c.H*c.T) (row : Fin (c.B*c.H)) : NativeIndex.PID c := by
  have bound := contract.grid (by rw [binding.numel]; exact nonempty)
  have rowCount : rows s=c.B*c.H := by simp [rows,binding.shape]
  exact ⟨row.val,row.isLt,by rw [rowCount] at bound; exact row.isLt.trans bound⟩

theorem aligned (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (c : Layout.Shape) (p : Storage.Placement c) (binding : TensorBinding s c p)
    (nonempty : 0<s.numel) (offset : Nat) :
    Storage.address p.backing.input offset % 4=0 := by
  simp [Storage.address,binding.inputPointer nonempty,Nat.add_mod,contract.alignment nonempty]
end TensorBinding

#axiomsClean checkInput_iff
#axiomsClean TensorBinding.input_read
#axiomsClean TensorBinding.physical_input_read
#axiomsClean TensorBinding.logical_input_read
#axiomsClean TensorBinding.pid
#axiomsClean TensorBinding.aligned
end VeriTile.Bench.Repairs.ChunkCumsumKernel
