import bench.tritonbench_g.triton_mul2.Proofs.Rewrite

/-! Operational native-memory facts, separate from the public equivalence. -/
namespace VeriTile.Bench.Repairs.TritonMul2.FPEquiv
open VeriTile.Triton
noncomputable section

/-- The complete FP source leaves every non-output or masked-out cell alone,
independently of whether input values satisfy a numerical rewrite domain. -/
theorem original_memory_frame {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (X O : RegionName) (n B : Nat) (s : FP.Structural.State α)
    (r : RegionName) (address : Nat)
    (outside : r ≠ O ∨ ∀ i : TileIndex [B], s.pids 0 * B + i.1.val < n →
      address ≠ s.pids 0 * B + i.1.val) :
    (Rewrite.originalExecution M X O n B s).val.mem r address = s.mem r address :=
  Rewrite.original_frame M X O n B s r address outside

#axiomsClean original_memory_frame
end
end VeriTile.Bench.Repairs.TritonMul2.FPEquiv
