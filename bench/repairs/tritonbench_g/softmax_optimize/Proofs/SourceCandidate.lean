import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only connection to the exact former AST construction. The actual
candidate is a complete ordinary quotation in Kernels.lean. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize
open VeriTile.Triton

def commutedOffset (B : Nat) : ComputeStmt :=
  .assign .int [B] "offset" (.alg (.add .int .scalarL
    (.mul .int .nil (.ref .int [] "cols") (.ref .int [] "pid_m"))
    (.ref .int [B] "n_offsets")))

def commuteLoop (B : Nat) : ComputeStmt → ComputeStmt
  | .forRangeInt idx start stop stride body =>
      .forRangeInt idx start stop stride (body.take 1 ++ [commutedOffset B] ++ body.drop 2)
  | statement => statement


namespace ReadableSource
def legacyIndexCommutedKernel (Y X : RegionName) (M N B : Nat) : ComputeKernel :=
  .mk [X] [Y] ((kernel Y X M N B).surfaceBody.map (commuteLoop B))

/-- Equality of complete source functions for every port, size and width. -/
theorem candidate_preserved : indexCommutedKernel=legacyIndexCommutedKernel := rfl
#axiomsClean candidate_preserved
end ReadableSource
end VeriTile.Bench.Repairs.SoftmaxOptimize
