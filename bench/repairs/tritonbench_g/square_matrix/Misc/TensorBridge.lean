import bench.repairs.tritonbench_g.square_matrix.Proofs.Indices
import bench.repairs.tritonbench_g.square_matrix.Correct
import bench.repairs.tritonbench_g.square_matrix.FPEquiv

namespace VeriTile.Bench.Repairs.SquareMatrix.FPEquiv
noncomputable section

/-- Source FP32 multiplication, with no assertion of exact Real arithmetic
or extra native-to-IEEE conformance experiment. -/
abbrev reference := @FPProof.reference

end
end VeriTile.Bench.Repairs.SquareMatrix.FPEquiv

/-! Optional Python checks and actual-Tensor pointer correspondence. -/
namespace VeriTile.Bench.Repairs.SquareMatrix.TensorBridge
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open Correct FPEquiv
open scoped VeriTile.Spec VeriTile.Triton.MaskedKernelIO₁
open VeriTile.Torch.FP32
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := checkInput_iff x

/-- The checked finite pointer expression reads the logical Tensor element,
including nonzero storage offsets and irrelevant singleton strides. -/
theorem square_input_address (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (read : Nat → Word) (row col : Nat)
    (rowBound : row < rows x.metadata) (active : col < cols x.metadata) :
    read ((Indices.pointer x.dataPtr (cols x.metadata) row col).toNat) =
      logicalValue read x laws (row*cols x.metadata+col) :=
  Indices.tensor_pointer_read x laws h read row col rowBound active

/-- One complete launched row: independent elementwise square, successful
flat execution, active access bounds and the exterior memory frame.
Empty inputs return before launch and create no kernel obligation. -/
theorem square_kernel_correctness (x : TensorMetadata) (_h : InputContract x)
    (_nonempty : 0 < x.numel) :
    Spec.Real (squareIO (cols x) (block (cols x)) ⊨
      fun xs i => mathematical (xs i)) := KernelReal.implements _ _

/-- Every logical column of this program's row is covered, including a padded
last lane group. The target is pointwise square, not a matrix product. -/
theorem square_kernel_coordinates (x : TensorMetadata) (_h : InputContract x)
    (_nonempty : 0 < x.numel) (s : BlockState) (_row : s.pids 0 < rows x) :
    Spec.Real (∃ t, exec ((fp32Kernel (cols x) (block (cols x))).eraseDType.toAlgKernel) s = some t ∧
      (∀ col, col < cols x → t.readMem "out" (s.pids 0*cols x+col) =
        mathematical (s.readMem "x" (s.pids 0*cols x+col))) ∧
      (∀ (r : RegionName) o,
        r ≠ "out" ∨ o < s.pids 0*cols x ∨ s.pids 0*cols x+cols x ≤ o →
        t.mem r o = s.mem r o)) := KernelReal.coordinates (cols x) s


/-- Both actual sources succeed and agree on every typed memory cell. Only
integer index multiplication is commuted; FP32 multiplication is identical.
There are no finite-value or inactive-lane numerical requirements. -/
theorem square_index_equiv (x : TensorMetadata) (_h : InputContract x)
    (_nonempty : 0 < x.numel) :
    original (cols x) (block (cols x)) ≡[assumptions]
      indexCommuted (cols x) (block (cols x)) :=
  FPEquiv.square_index_equiv (cols x) (block (cols x))

/-- Kernel execution is tied to the original Tensor's actual storage values.
All FP32 words remain admitted. The result covers every logical column of
this row; all other typed memory, including every input cell, is preserved. -/
theorem square_kernel_fp32 (M : Algebra Word)
    (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (_nonempty : 0 < x.metadata.numel) (s : State Word)
    (row : Nat) (rowBound : row < rows x.metadata) (pid : s.pids 0 = row)
    (read : Nat → Word) (input : InputCells x laws h read s.mem) :
    ∃ t, exec (FPProof.backend M) (fp32Kernel (cols x.metadata) (block (cols x.metadata))) s = some t ∧
      (∀ col, col < cols x.metadata → t.mem "out" (row*cols x.metadata+col) =
        .mk .real (reference M (logicalValue read x laws (row*cols x.metadata+col)))) ∧
      (∀ (r : RegionName) o,
        r ≠ "out" ∨ o < row*cols x.metadata ∨ row*cols x.metadata+cols x.metadata ≤ o →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := KernelFP.coordinates M (cols x.metadata) s
  refine ⟨t, he, ?_, ?_⟩
  · intro col active
    have hi : row*cols x.metadata+col < x.metadata.numel := by
      rw [numel_eq x.metadata h]
      exact flat_lt _ _ _ _ rowBound active
    have value := hv col active
    simp only [pid] at value
    rw [input _ hi, Cell.read_same, direct_read_eq_logical read x laws h.contiguous h.noNeg _ hi] at value
    exact value
  · simpa only [pid] using hf

#axiomsClean square_kernel_correctness
#axiomsClean square_kernel_coordinates
#axiomsClean square_index_equiv
#axiomsClean square_kernel_fp32

#axiomsClean python_checks_iff_preconditions
#axiomsClean square_input_address
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.TensorBridge
