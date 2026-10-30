import bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSource
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

abbrev BHW (p : Parameters) := p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH
abbrev BIN (p : Parameters) := p.BLOCK_SIZE_IN_FEAT
abbrev OF (p : Parameters) := p.BLOCK_SIZE_OUT_FEAT
abbrev AIndex (p : Parameters) := TileIndex [BHW p,BIN p]
abbrev BIndex (p : Parameters) := TileIndex [BIN p,OF p]
abbrev CIndex (p : Parameters) := TileIndex [BHW p,OF p]

def prelude (p : Parameters) := (source p).surfaceBody.take 14
def hBody (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[14]? with
  | some (ComputeStmt.forLoop _ _ code) => code
  | _ => []
def wBody (p : Parameters) : List ComputeStmt :=
  match (hBody p)[0]? with
  | some (ComputeStmt.forLoop _ _ code) => code
  | _ => []
def cBody (p : Parameters) : List ComputeStmt :=
  match (wBody p)[0]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 15

/-- Typed fp32-input, fp32-output dot with all original options visible. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : Dot α) : Algebra α :=
  {rest with dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
    match left,right,out with
    | .fp32,.fp32,.fp32 => some (D.apply precision options a b)
    | l,r,o => rest.dotTyped precision l r o options a b}

theorem decomposition (p : Parameters) :
    (source p).surfaceBody = prelude p ++
      [.forLoop "h" p.kernel_height (hBody p)] ++ epilogue p := by rfl

theorem hBody_shape (p : Parameters) :
    hBody p = [.forLoop "w" p.kernel_width (wBody p)] := by rfl

theorem wBody_shape (p : Parameters) :
    wBody p = [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim")
      (.constNat (BIN p)) (cBody p)] := by rfl

#axiomsClean decomposition
#axiomsClean hBody_shape
#axiomsClean wBody_shape
end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault
