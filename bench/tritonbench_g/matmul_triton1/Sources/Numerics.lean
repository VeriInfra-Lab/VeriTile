import bench.tritonbench_g.matmul_triton1.Sources.Parameters
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
open VeriTile.Triton FP.Structural
noncomputable section
abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
structure Primitives (α : Type) where
  zero : α
  dot : {k : Nat} → (Fin k → α) → (Fin k → α) → α → α

def matrixDot {α : Type} (P : Primitives α)
    {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → α) (b : TileIndex (batch ++ [k,n]) → α)
    (acc : TileIndex (batch ++ [m,n]) → α) : TileIndex (batch ++ [m,n]) → α :=
  match batch with
  | [] => fun i => P.dot (fun t => a (i.1,t,PUnit.unit))
      (fun t => b (t,i.2.1,PUnit.unit)) (acc i)
  | _::tail => fun i => matrixDot P (batch:=tail) (fun j => a (i.1,j))
      (fun j => b (i.1,j)) (fun j => acc (i.1,j)) i.2
termination_by batch.length

end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core
