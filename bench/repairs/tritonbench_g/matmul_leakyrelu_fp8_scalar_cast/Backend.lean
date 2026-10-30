import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural

/-- Task-local operation interface, rebuilt against the current shared Algebra
record. This avoids unfolding the frozen original cache's older record layout. -/
structure DotAcc (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α) → (TileIndex (batch ++ [m,n]) → α)
  le : Option ComputeDType → FloatDType → α → α → Bool

def backend {α : Type} (rest : Algebra α) (D : DotAcc α) : Algebra α :=
  {rest with
    compareLe := fun p d => some (D.le p d)
    dotAcc := fun p l r o opt {batch} {m} {k} {n} a b c =>
      some (fun i => ofFloat o (D.apply p l r o opt
        (fun j => toFloat l (a j)) (fun j => toFloat r (b j))
        (fun j => toFloat o (c j)) i))}

end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
