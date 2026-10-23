import bench.repairs.tritonbench_g.chunk_cumsum_vector.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Storage
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkCumsumVector
open VeriTile.Triton VeriTile.Torch
open scoped BigOperators

theorem contract_iff (s : TensorMetadata) (dtype : Option DType) :
    InputContract s dtype ↔ s.layout = .strided ∧ s.dtype = .floating .fp32 ∧
      (dtype = none ∨ dtype = some (.floating .fp32)) ∧ s.shape.length = 4 ∧
      s.device.kind = "cuda" ∧ s.isContiguous = Bool.true ∧ s.isNeg = Bool.false ∧
      (0 < s.numel → rows s ≤ 65535 ∧ featureBlocks s < 2^31) ∧ (0 < s.numel → s.dataPtr.toNat % 4 = 0) := by
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
  shape : s.shape = [h.B,h.H,h.T,h.S]
  inputDType : h.inputDType = .fp32
  outputDType : h.outputDType = .fp32
  inputPointer : 0 < s.numel → p.backing.input.base = s.dataPtr.toNat
  inputFlat : ∀ row time channel, row < h.B*h.H → time < h.T → channel < h.S →
    h.inputStrides.address h.H row time channel = (row*h.T+time)*h.S+channel
  outputFlat : ∀ row time channel, row < h.B*h.H → time < h.T → channel < h.S →
    h.outputStrides.address h.H row time channel = (row*h.T+time)*h.S+channel
  inputBytes : p.backing.input.bytes = 4
  outputBytes : p.backing.output.bytes = 4
  distinct : h.input ≠ h.output
  outputAlignment : 0 < s.numel → p.backing.output.base % 4 = 0

/-- Ordinary contiguous Tensor input readback from the current physical heap.
This law states only input representation; it mentions no kernel or output.
The same law is used with Real cells and with typed FP payload cells. -/
structure InputReadLaw {β γ : Type} (s : TensorMetadata) (h : Call)
    (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → Nat → γ) : Prop where
  reads : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
    read (heap (s.dataPtr.toNat+((row*h.T+time)*h.S+channel)*4))=logical row time channel

namespace TensorBinding
theorem input_read {β : Type} (s : TensorMetadata) (h : Call)
    (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<s.numel) (other : β) (heap : Nat → β) (offset : Nat) :
    Storage.lift h p other heap h.input offset = heap (s.dataPtr.toNat+offset*4) := by
  rw [Storage.lift_input]
  simp only [Storage.address,binding.inputPointer nonempty,binding.inputBytes]

private theorem block_unique (n a b i j : Nat) (hi : i<n) (hj : j<n)
    (eq : a*n+i=b*n+j) : a=b ∧ i=j := by
  have same : a=b := by
    rcases lt_trichotomy a b with less | same | greater
    · have gap := Nat.mul_le_mul_right n (Nat.succ_le_of_lt less)
      rw [Nat.succ_mul] at gap
      omega
    · exact same
    · have gap := Nat.mul_le_mul_right n (Nat.succ_le_of_lt greater)
      rw [Nat.succ_mul] at gap
      omega
  subst b
  exact ⟨rfl,by omega⟩

theorem outputLayout (s : TensorMetadata) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) : Geometry.OutputLayout h := by
  refine ⟨binding.distinct,?_⟩
  intro r q t u x y hr hq ht hu hx hy eq
  rw [binding.outputFlat r t x hr ht hx,binding.outputFlat q u y hq hu hy] at eq
  obtain ⟨same,xy⟩ := block_unique h.S _ _ _ _ hx hy eq
  obtain ⟨rq,tu⟩ := block_unique h.T _ _ _ _ ht hu same
  exact ⟨rq,tu,xy⟩

theorem numel (s : TensorMetadata) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) : s.numel=h.B*h.H*h.T*h.S := by
  simp [TensorMetadata.numel,binding.shape,Nat.mul_assoc]

theorem physical_input_read {β γ : Type} (s : TensorMetadata) (h : Call)
    (p : Storage.Placement h) (binding : TensorBinding s h p)
    (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (law : InputReadLaw s h heap read logical)
    (row time channel : Nat) (hr : row<h.B*h.H) (ht : time<h.T) (hc : channel<h.S) :
    read (heap (Storage.address p.backing.input
      (h.inputStrides.address h.H row time channel)))=logical row time channel := by
  have nonempty : 0<s.numel := by
    rw [binding.numel]
    exact Nat.mul_pos (Nat.mul_pos (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt ht))
      (Nat.zero_lt_of_lt hc)
  simpa only [Storage.address,binding.inputPointer nonempty,binding.inputBytes,
    binding.inputFlat row time channel hr ht hc] using law.reads row time channel hr ht hc

theorem logical_input_read {β γ : Type} (s : TensorMetadata) (h : Call)
    (p : Storage.Placement h) (binding : TensorBinding s h p)
    (other : β) (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (law : InputReadLaw s h heap read logical)
    (row time channel : Nat) (hr : row<h.B*h.H) (ht : time<h.T) (hc : channel<h.S) :
    read (Storage.lift h p other heap h.input
      (h.inputStrides.address h.H row time channel))=logical row time channel := by
  rw [Storage.lift_input]
  exact physical_input_read s h p binding heap read logical law row time channel hr ht hc

def pid (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<h.B*h.H*h.T*h.S) (prog : Geometry.Program h) : NativeIndex.PID h := by
  have bound := contract.grid (by rw [binding.numel]; exact nonempty)
  have rowCount : rows s=h.B*h.H := by simp [rows,binding.shape]
  have featureCount : featureBlocks s=Geometry.features h := by
    simp [featureBlocks,binding.shape,Geometry.features,BS]
  refine ⟨prog,?_,?_⟩
  · rw [featureCount] at bound
    exact prog.featureInside.trans bound.2
  · rw [rowCount] at bound
    have := prog.rowInside
    omega

theorem aligned (s : TensorMetadata) (dtype : Option DType) (contract : InputContract s dtype)
    (h : Call) (p : Storage.Placement h) (binding : TensorBinding s h p)
    (nonempty : 0<s.numel) (offset : Nat) :
    Storage.address p.backing.input offset % 4 = 0 := by
  simp [Storage.address,binding.inputBytes,binding.inputPointer nonempty,Nat.add_mod,contract.alignment nonempty]
end TensorBinding

#axiomsClean checkInput_iff
#axiomsClean TensorBinding.input_read
#axiomsClean TensorBinding.physical_input_read
#axiomsClean TensorBinding.logical_input_read
#axiomsClean TensorBinding.outputLayout
#axiomsClean TensorBinding.pid
#axiomsClean TensorBinding.aligned
end VeriTile.Bench.Repairs.ChunkCumsumVector
