import bench.tritonbench_g.matmul_leakyrelu.CandidateSourceDefinitions
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

def tripExpr (p : Parameters) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat (BK p))) (.constNat 1)) (.constNat (BK p))
def prelude (p : Parameters) (st : Storage) := (source p st).surfaceBody.take 15
def body (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (source p st).surfaceBody[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) (st : Storage) := (source p st).surfaceBody.drop 16

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

theorem source_decomposition (p : Parameters) (st : Storage) :
    (source p st).surfaceBody = prelude p st ++
      [.forRangeDyn "k" (.constNat 0) (tripExpr p) (.constNat 1) (body p st)] ++ epilogue p st := by
  cases st <;> rfl
#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
