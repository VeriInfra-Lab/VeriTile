import bench.examples.Welford.Contract

namespace VeriTile.Bench.Examples.LayerNormFPContract

open VeriTile Triton
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)

/-- Nonempty rows use Welford's domain for initialization, each update and
both normalization paths. Empty rows require no numerical domain checks. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : FP.GuardExpression.Condition :=
  if N = 0 then .top else WelfordFPEquiv.requirements x N stride empty plans

end VeriTile.Bench.Examples.LayerNormFPContract

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv

abbrev requirements := LayerNormFPContract.requirements

end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
