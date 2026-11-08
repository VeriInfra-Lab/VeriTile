import bench.tritonbench_g.matmul_triton_autotune.Kernels
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.Sources.Mathematical_TypedCopy
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- Compatibility call wrapper for the complete repaired Python body. -/
def typed (dtype : FloatDType) (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) (ACTIVATION : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
    BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M
    (if ACTIVATION then "leaky_relu" else "")

def source (p : Parameters) (st : Storage) : ComputeKernel :=
  typed (dtype st) p.A p.B p.C p.M p.N p.K p.stride_am p.stride_ak p.stride_bk p.stride_bn p.stride_cm p.stride_cn p.BLOCK_SIZE_M p.BLOCK_SIZE_N p.BLOCK_SIZE_K p.GROUP_SIZE_M p.ACTIVATION
abbrev BM (p : Parameters) := p.BLOCK_SIZE_M
abbrev BN (p : Parameters) := p.BLOCK_SIZE_N
abbrev BK (p : Parameters) := p.BLOCK_SIZE_K
abbrev AIndex (p : Parameters) := TileIndex [BM p,BK p]
abbrev BIndex (p : Parameters) := TileIndex [BK p,BN p]
abbrev CIndex (p : Parameters) := TileIndex [BM p,BN p]
def prelude (p : Parameters) (st : Storage) := (source p st).surfaceBody.take 22
def body (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (source p st).surfaceBody[22]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) (st : Storage) := (source p st).surfaceBody.drop 23

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

def loopStatement (p : Parameters) (st : Storage) : ComputeStmt :=
  ((source p st).surfaceBody[22]?).getD (.effectMarker "missing source loop")
theorem source_decomposition (p : Parameters) (st : Storage) :
    (source p st).surfaceBody = prelude p st ++ [loopStatement p st] ++ epilogue p st := by
  cases st <;> rfl

#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
