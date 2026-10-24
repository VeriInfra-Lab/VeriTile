import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Machine
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LogsumexpFwd.KernelBinding
open VeriTile.Triton FP.Structural
noncomputable section

/-- The admitted descriptor is bound to the actual Tensor and its storage.
These ordinary API laws contain no kernel output or arithmetic assumption. -/
structure Input (a : Arguments) where
  tensor : VeriTile.Torch.FP32.Tensor
  metadata : tensor.metadata = a.input
  laws : VeriTile.Torch.FP32.TensorLaws tensor

theorem Input.contiguous (x : Input a) (h : InputContract a) :
    x.tensor.metadata.isContiguous = Bool.true := by
  rw [x.metadata]
  exact h.2.2.2.2.1

theorem Input.noNeg (x : Input a) (h : InputContract a) :
    x.tensor.metadata.isNeg = Bool.false := by
  rw [x.metadata]
  exact h.2.2.2.2.2.1

def Input.view (x : Input a) (h : InputContract a) : VeriTile.Torch.FP32.DataView :=
  VeriTile.Torch.FP32.dataView x.tensor x.laws (x.contiguous h)

/-- Logical coordinates of the exact-Real interpretation use the same
physical Tensor indices. The public contract excludes delayed negation. -/
def Input.realValue (x : Input a) (read : Nat → ℝ) (i : Nat) : ℝ :=
  read (x.tensor.storage.address (x.tensor.physicalIndex i))

def Input.wordValue (x : Input a) (read : Nat → VeriTile.Torch.FP32.Word) (i : Nat) : BitVec 32 :=
  (VeriTile.Torch.FP32.logicalValue read x.tensor x.laws i).bits

theorem Input.real_direct (x : Input a) (h : InputContract a) (read : Nat → ℝ)
    (i : Nat) (hi : i < a.input.numel) :
    read ((x.view h).address i) = x.realValue read i := by
  have inside : i < x.tensor.metadata.numel := by simpa only [x.metadata] using hi
  unfold Input.realValue
  rw [x.laws.contiguous (x.contiguous h) i inside]
  apply congrArg read
  simp only [Input.view, VeriTile.Torch.FP32.dataView, VeriTile.Torch.FP32.DataView.address,
    VeriTile.Torch.FP32.DataView.base, if_neg (by omega : x.tensor.metadata.numel ≠ 0),
    VeriTile.Torch.FP32.Storage.address]
  omega

theorem Input.word_direct (x : Input a) (h : InputContract a)
    (read : Nat → VeriTile.Torch.FP32.Word) (i : Nat) (hi : i < a.input.numel) :
    (read ((x.view h).address i)).bits = x.wordValue read i := by
  apply congrArg (fun v : VeriTile.Torch.FP32.Word => v.bits)
  exact VeriTile.Torch.FP32.direct_read_eq_logical read x.tensor x.laws (x.contiguous h)
    (x.noNeg h) i (by simpa only [x.metadata] using hi)

def RealCells (x : Input a) (h : InputContract a) (read : Nat → ℝ)
    (s : State (WithBot ℝ)) : Prop :=
  ∀ i, i < a.input.numel → s.mem "input" i =
    .mk .real (read ((x.view h).address i) : WithBot ℝ)

def WordCells (x : Input a) (h : InputContract a) (read : Nat → VeriTile.Torch.FP32.Word)
    (s : State (BitVec 32)) : Prop :=
  ∀ i, i < a.input.numel → s.mem "input" i =
    .mk .real (read ((x.view h).address i)).bits

theorem real_cells (x : Input a) (h : InputContract a) (read : Nat → ℝ)
    (s : State (WithBot ℝ)) (input : RealCells x h read s)
    (i : Nat) (hi : i < a.input.numel) :
    (s.mem "input" i).read .real = (x.realValue read i : WithBot ℝ) := by
  rw [input i hi, ←x.real_direct h read i hi]
  rfl

theorem word_cells (x : Input a) (h : InputContract a) (read : Nat → VeriTile.Torch.FP32.Word)
    (s : State (BitVec 32)) (input : WordCells x h read s)
    (i : Nat) (hi : i < a.input.numel) :
    (s.mem "input" i).read .real = x.wordValue read i := by
  rw [input i hi, ←x.word_direct h read i hi]
  rfl

/-- The finite pointer is the same observed data_ptr used by Python admission,
including nonzero storage offsets. Inactive lanes need no memory contents. -/
theorem input_pointer (x : Input a) (h : InputContract a) (i : Nat)
    (hi : i < a.input.numel) :
    (a.input.dataPtr + BitVec.ofNat 64 (4*i)).toNat = (x.view h).address i := by
  have nonempty : 0 < x.tensor.metadata.numel := by rw [x.metadata]; omega
  have pointer := VeriTile.Torch.FP32.actual_pointer x.tensor x.laws (x.contiguous h) nonempty
  change BitVec.ofNat 64 ((x.view h).address 0) = x.tensor.metadata.dataPtr at pointer
  rw [x.metadata] at pointer
  rw [←pointer, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat]
  have span := (x.view h).spanFits
  have elements : (x.view h).elements = a.input.numel := by
    simp only [Input.view,VeriTile.Torch.FP32.dataView,x.metadata]
  rw [elements] at span
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero]
  exact Nat.mod_eq_of_lt (by omega)

theorem input_aligned (x : Input a) (h : InputContract a) :
    (x.view h).base % 4 = 0 := by
  apply VeriTile.Torch.FP32.dataView_aligned x.tensor x.laws (x.contiguous h)
  intro nonempty
  change x.tensor.metadata.dataPtr.toNat % 4 = 0
  rw [x.metadata]
  exact h.2.2.2.2.2.2.2.1 (by simpa only [x.metadata] using nonempty)

/-- Scratch is an ordinary freshly allocated FP32 Tensor. The original
Torch aggregation consumes this (rows, blocks) buffer outside kernel scope. -/
structure Output (a : Arguments) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape = [a.rows, a.blocks]
  contiguous : tensor.metadata.isContiguous = Bool.true
  noNeg : tensor.metadata.isNeg = Bool.false
  device : tensor.metadata.device = a.input.device
  aligned : 0 < tensor.metadata.numel → tensor.dataPtr.toNat % 4 = 0

def Output.view (z : Output a) : VeriTile.Torch.FP32.DataView :=
  VeriTile.Torch.FP32.dataView z.tensor z.laws z.contiguous

theorem Output.elements (z : Output a) : z.view.elements = a.rows*a.blocks := by
  simp [Output.view,VeriTile.Torch.FP32.dataView,VeriTile.Torch.TensorMetadata.numel,z.shape]

theorem output_pointer (z : Output a) (i : Nat) (hi : i < a.rows*a.blocks) :
    (z.tensor.metadata.dataPtr + BitVec.ofNat 64 (4*i)).toNat = z.view.address i := by
  have size : z.tensor.metadata.numel = a.rows*a.blocks := by
    simpa only [Output.view,VeriTile.Torch.FP32.dataView] using z.elements
  have pointer := VeriTile.Torch.FP32.actual_pointer z.tensor z.laws z.contiguous (by rw [size]; omega)
  change BitVec.ofNat 64 (z.view.address 0) = z.tensor.metadata.dataPtr at pointer
  rw [←pointer,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have span := z.view.spanFits
  rw [z.elements] at span
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero]
  exact Nat.mod_eq_of_lt (by omega)

#axiomsClean real_cells
#axiomsClean word_cells
#axiomsClean input_pointer
#axiomsClean input_aligned
#axiomsClean output_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.KernelBinding
