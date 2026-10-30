import bench.tritonbench_g.lightning_attention.Parts
import bench.tritonbench_g.lightning_attention.Proofs.NativeMask
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
/-- Internal finite constant-mask facts; public interfaces instantiate them with
proved Real/IEEE operations. There is no payload arithmetic assumption here. -/
structure MaskArithmetic {α : Type} (M : Algebra α) where
  le : α → α → Bool
  compare : M.compareLe (some .fp32) .real = some le
  lower : ∀ i j : Nat, i < 64 → j < 64 →
    le (zero M) (M.binary (some .fp32) .real .sub (M.fromNat (some .fp32) i)
      (M.fromNat (some .fp32) j)) = decide (j ≤ i)
  upper : ∀ i j : Nat, i < 64 → j < 64 →
    le (M.binary (some .fp32) .real .sub (M.fromNat (some .fp32) i)
      (M.fromNat (some .fp32) j)) (zero M) = decide (i ≤ j)
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
