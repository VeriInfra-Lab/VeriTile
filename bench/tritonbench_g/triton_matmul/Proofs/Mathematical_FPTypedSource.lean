import bench.tritonbench_g.triton_matmul.Sources.Mathematical_SourceDefinitions
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

structure DotAcc (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α) → (TileIndex (batch ++ [m,n]) → α)
def backend {α : Type} (rest : Algebra α) (D : DotAcc α) : Algebra α :=
  {rest with dotAcc := fun p l r o opt {batch} {m} {k} {n} a b c =>
    match l,r,o with
    | .fp16,.fp16,.real => some (D.apply p .fp16 .fp16 .real opt a b c)
    | .f8e4,.f8e4,.real => some (D.apply p .f8e4 .f8e4 .real opt a b c)
    | l,r,o => rest.dotAcc p l r o opt a b c}

theorem source_decomposition (p : Parameters) (st : Storage) :
    (source p st).surfaceBody = prelude p st ++
      [.forRange "kk" 0 p.numKBlocks 1 (body p st)] ++ epilogue p st := by
  cases st <;> rfl
#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
