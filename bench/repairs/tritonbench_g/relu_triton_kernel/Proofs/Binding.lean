import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Contract
import VeriTile.Torch.FP32

/-! Kernel-input binding only. These theorems connect checked metadata with
actual Tensor storage and direct pointer reads; they do not execute a wrapper. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.Binding
open VeriTile.Triton
open VeriTile.Torch.FP32
noncomputable section

/-- Contiguity and the checked negative-view flag identify the kernel's direct
FP32 reads with logical input elements, including nonzero storage offsets. -/
theorem input_read (read : Nat → Word) (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    read ((dataView x laws h.contiguous).address i) = logicalValue read x laws i :=
  direct_read_eq_logical read x laws h.contiguous h.noNeg i hi

/-- The logical view starts at the Python-observed pointer. -/
theorem input_pointer (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (nonempty : 0 < x.metadata.numel) :
    BitVec.ofNat 64 ((dataView x laws h.contiguous).address 0) = x.metadata.dataPtr :=
  actual_pointer x laws h.contiguous nonempty

/-- Python's alignment guard supplies the scalar FP32 access alignment. -/
theorem input_alignment (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) :
    (dataView x laws h.contiguous).base % 4 = 0 :=
  dataView_aligned x laws h.contiguous h.alignment

/-- Every live input address starts at the actual Python-observed pointer.
The storage span proves that converting the pointer to Nat does not wrap. -/
theorem input_address (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    (dataView x laws h.contiguous).address i = x.dataPtr.toNat + 4*i := by
  have nonempty : 0 < x.metadata.numel := by omega
  have ptr := input_pointer x laws h nonempty
  have span := (dataView x laws h.contiguous).spanFits
  have bound : (dataView x laws h.contiguous).base < 2^64 := by
    change (dataView x laws h.contiguous).base + 4*x.metadata.numel ≤ 2^64 at span
    omega
  have base : (dataView x laws h.contiguous).base = x.dataPtr.toNat := by
    simpa only [DataView.address, Nat.mul_zero, Nat.add_zero,
      BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound] using congrArg BitVec.toNat ptr
  simp only [DataView.address, base]

theorem pointer_read (read : Nat → Word) (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    read (x.dataPtr.toNat + 4*i) = logicalValue read x laws i := by
  rw [←input_address x laws h i hi]
  exact input_read read x laws h i hi

/-- Interpret the actual input words mathematically. `decode` is a supplied
mathematical interpretation, not an assertion that FP32 execution is exact
Real arithmetic. Other regions, registers and program IDs remain arbitrary. -/
def realState (x : Tensor) (read : Nat → Word) (decode : Word → ℝ)
    (surrounding : BlockState) : BlockState :=
  { surrounding with mem := fun r i =>
      if r = "x" then .real (decode (read (x.dataPtr.toNat + 4*i)))
      else surrounding.mem r i }

theorem realState_input (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (read : Nat → Word) (decode : Word → ℝ)
    (s : BlockState) (i : Nat) (hi : i < x.metadata.numel) :
    (realState x read decode s).readMem "x" i = decode (logicalValue read x laws i) := by
  change decode (read (x.dataPtr.toNat + 4*i)) = _
  rw [pointer_read read x laws h i hi]

/-- Cells outside this program's active output, including the input region. -/
def Outside (n pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ "out" ∨ ∀ i : Fin blockSize, pid*blockSize+i.val < n → o ≠ pid*blockSize+i.val

#axiomsClean input_read
#axiomsClean input_pointer
#axiomsClean input_alignment
#axiomsClean input_address
#axiomsClean pointer_read
#axiomsClean realState_input
end
end VeriTile.Bench.Repairs.ReluTritonKernel.Binding
