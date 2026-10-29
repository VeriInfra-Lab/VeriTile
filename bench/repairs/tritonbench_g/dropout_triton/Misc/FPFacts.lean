import bench.repairs.tritonbench_g.dropout_triton.Proofs.Rewrite
import bench.repairs.tritonbench_g.dropout_triton.Proofs.Typed

/-! Operational native-memory facts used by optional applications. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.FPEquiv
open VeriTile.Triton
noncomputable section

/-- Source-precision output formula under the standard same-dtype identity
cast interpretation; this fixes primitive semantics, not an experimental law.
The formula reads entry FP32 data and signed-int32 mask values. -/
theorem dropout_kernel_result {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (n : Nat) (p : ℝ) (before : FP.Structural.State α) :
    ∃ final, FP.Structural.exec (Typed.castBackend M) (kernel n p) before = some final ∧
      final.mem = Typed.expectedMemory M ⟨"x", "keep", "out", n, blockSize⟩ p before :=
  Typed.complete_correct M ⟨"x", "keep", "out", n, blockSize⟩ p before

theorem dropout_kernel_frame {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (n : Nat) (p : ℝ) (before : FP.Structural.State α) (r : RegionName) (address : Nat)
    (outside : r ≠ "out" ∨ ∀ i : TileIndex [blockSize],
      before.pids 0 * blockSize + i.1.val < n →
      address ≠ before.pids 0 * blockSize + i.1.val) :
    (Rewrite.originalExecution M n p before).val.mem r address = before.mem r address :=
  Rewrite.original_frame M n p before r address outside

#axiomsClean dropout_kernel_result
#axiomsClean dropout_kernel_frame
end
end VeriTile.Bench.Repairs.DropoutTriton.FPEquiv
