import bench.tritonbench_g.softmax_optimize.Kernels
import VeriTile.Meta.StatementAudit

/-! Proof-only decomposition of the current complete source pair.
The quoted candidate differs by exactly one integer commutation per loop. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize
open VeriTile.Triton

def commutedOffset (N B : Nat) : ComputeStmt :=
  .assign .int [B] "offset" (.alg (.add .int .scalarL
    (.mul .int .nil (.constInt N) (.ref .int [] "pid_m"))
    (.ref .int [B] "n_offsets")))

def commuteLoop (N B : Nat) : ComputeStmt → ComputeStmt
  | .forRangeInt idx start stop stride body =>
      .forRangeInt idx start stop stride (body.take 1 ++ [commutedOffset N B] ++ body.drop 2)
  | statement => statement


namespace ReadableSource
/-- A body decomposition, not a second authoritative kernel implementation. -/
theorem candidate_body (Y X : RegionName) (M N B : Nat) :
    (indexCommutedKernel Y X M N B).surfaceBody=
      (sourceKernel Y X M N B).surfaceBody.map (commuteLoop N B) := rfl
#axiomsClean candidate_body
end ReadableSource
end VeriTile.Bench.Repairs.SoftmaxOptimize
