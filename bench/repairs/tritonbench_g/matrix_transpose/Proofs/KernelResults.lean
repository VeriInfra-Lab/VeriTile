import bench.repairs.tritonbench_g.matrix_transpose.Proofs.KernelReal
import bench.repairs.tritonbench_g.matrix_transpose.Proofs.KernelFP

/-! Operational coordinate, native-word and pointer corollaries of the shared
execution foundations. Public IO entries remain independent of this module. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.KernelResults
open VeriTile.Triton FP.Structural
noncomputable section

/-- The independent formula is output[col,row] = input[row,col]. This public
coordinate statement exposes the admitted shape and reads the initial input. -/
theorem transpose_kernel_coordinates (m d : Nat) (h : InputContract m d)
    (initial : BlockState) :
    Spec.Real (∃ final,
      exec ((source "matrix" "out" m d).eraseDType.toAlgKernel) initial = some final ∧
      (∀ row : Fin m, ∀ col : Fin d,
        final.readMem "out" (col.val*m+row.val) =
          formula (initial.readMem "matrix") d row.val col.val) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ m*d ≤ o → final.mem r o = initial.mem r o)) :=
  KernelReal.coordinates m d h initial

/-- All 16 storage bits are copied to the independently specified transposed
coordinate. NaN payloads, signed zeros, infinities and subnormals are admitted.
The exterior frame includes every input cell and every non-output region. -/
theorem transpose_kernel_fp16_correct (m d : Nat) (h : InputContract m d)
    (initial : State (BitVec 16)) (matrix : Nat → BitVec 16)
    (input : InputCells m d initial.mem matrix) (M : Algebra (BitVec 16)) :
    ∃ final, exec M (source "matrix" "out" m d) initial = some final ∧
      (∀ row : Fin m, ∀ col : Fin d,
        final.mem "out" (col.val*m+row.val) = .mk .fp16 (formula matrix d row.val col.val)) ∧
      (∀ (r : RegionName) o, r ≠ "out" ∨ m*d ≤ o → final.mem r o = initial.mem r o) :=
  KernelFP.coordinates M m d h initial matrix input

/-- The complete typed source is independent of floating primitives. -/
theorem transpose_no_numerical_assumptions (m d : Nat) (s : State (BitVec 16))
    (left right : Algebra (BitVec 16)) :
    exec left (source "matrix" "out" m d) s =
      exec right (source "matrix" "out" m d) s := KernelFP.backend_independent left right m d s

/-- Exact source-order uint64 input/output pointers under ordinary fresh,
aligned allocation bounds. Every unmasked FP16 access stays in its window. -/
theorem transpose_source_addresses (m d : Nat) (h : InputContract m d)
    (inputBase outputBase : BitVec 64)
    (inputBound : inputBase.toNat + 2*(m*d) ≤ 2^64)
    (outputBound : outputBase.toNat + 2*(m*d) ≤ 2^64)
    (inputAligned : inputBase.toNat % 2 = 0)
    (outputAligned : outputBase.toNat % 2 = 0)
    (row : Fin m) (col : Fin d) :
    ((Indices.pointer64 inputBase row.val d col.val).toNat = inputBase.toNat+2*(row.val*d+col.val) ∧
      (Indices.pointer64 inputBase row.val d col.val).toNat % 2 = 0 ∧
      inputBase.toNat ≤ (Indices.pointer64 inputBase row.val d col.val).toNat ∧
      (Indices.pointer64 inputBase row.val d col.val).toNat+1 < inputBase.toNat+2*(m*d)) ∧
    ((Indices.pointer64 outputBase col.val m row.val).toNat = outputBase.toNat+2*(col.val*m+row.val) ∧
      (Indices.pointer64 outputBase col.val m row.val).toNat % 2 = 0 ∧
      outputBase.toNat ≤ (Indices.pointer64 outputBase col.val m row.val).toNat ∧
      (Indices.pointer64 outputBase col.val m row.val).toNat+1 < outputBase.toNat+2*(m*d)) := by
  have swapped : InputContract d m :=
    ⟨h.positive.symm, h.powers.symm, by simpa [Nat.mul_comm] using h.extent⟩
  exact ⟨Indices.allocation_access inputBase m d h row col inputBound inputAligned,
    by simpa [Nat.mul_comm] using (Indices.allocation_access outputBase d m swapped col row
      (by simpa [Nat.mul_comm] using outputBound) outputAligned)⟩

#axiomsClean transpose_kernel_coordinates
#axiomsClean transpose_kernel_fp16_correct
#axiomsClean transpose_no_numerical_assumptions
#axiomsClean transpose_source_addresses
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.KernelResults
