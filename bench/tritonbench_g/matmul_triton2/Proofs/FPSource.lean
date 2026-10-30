import bench.tritonbench_g.matmul_triton1.Proofs.FPSource
import bench.tritonbench_g.matmul_triton2.Sources.Kernels
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import VeriTile.Triton.Float.ScalarBackend
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
def prelude (p : Parameters) := (source p).surfaceBody.take 15
def loopStmt (p : Parameters) := (source p).surfaceBody[15]'(by simp [source,typed,ComputeKernel.surfaceBody])
def body (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 16

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [loopStmt p] ++ epilogue p := by rfl

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
abbrev Primitives := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.Core.Primitives
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

/-- `.real` is the frontend's FP32 payload channel here; the carrier can be
raw `FP.Value .fp32`. Loads preserve those payloads, the zero literal is local,
and accumulator-fed dot remains a distinct operation. -/
def backend {α : Type} (rest : Algebra α) (P : Primitives α) : Algebra α :=
  {rest with
    fp32Load := id
    literal := fun precision dtype value => if (dtype=.fp32 ∨ dtype=.real) ∧ value=0 then P.zero
      else rest.literal precision dtype value
    cast := fun precision src dst a => if (src=.fp32 ∨ src=.real) ∧ dst=.fp32 then a
      else rest.cast precision src dst a
    dotAcc := fun precision left right out options {batch} {m} {k} {n} a b acc =>
      match left,right,out with
      | .real,.real,.real => some (matrixDot P (batch:=batch) (m:=m) (k:=k) (n:=n) a b acc)
      | left,right,out => rest.dotAcc precision left right out options a b acc}


#axiomsClean source_decomposition
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.Core
