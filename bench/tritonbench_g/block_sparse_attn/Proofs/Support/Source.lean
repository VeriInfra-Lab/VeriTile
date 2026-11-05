import bench.tritonbench_g.block_sparse_attn.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def prelude (p : Parameters) := (source p).surfaceBody.take 27
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody[27]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeInt _ _ _ _ statements) => statements
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 28
def scoreBody (p : Parameters) : List ComputeStmt :=
  match ((body p)[3]? : Option ComputeStmt) with
  | some (ComputeStmt.forLoop _ _ statements) => statements
  | _ => []

theorem partition (p : Parameters) :
    (source p).surfaceBody = prelude p ++
      [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
        (.ref .int [] "end_l") (.constNat 1) (body p)] ++ epilogue p := by
  rfl

theorem score_partition (p : Parameters) :
    body p = (body p).take 3 ++ [.forLoop "score_block" p.blocks (scoreBody p)] ++
      (body p).drop 4 := by
  rfl

#axiomsClean partition
#axiomsClean score_partition
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
