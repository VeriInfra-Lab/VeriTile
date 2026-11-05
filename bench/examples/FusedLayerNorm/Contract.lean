import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import bench.examples.FusedLayerNorm.Kernels
import bench.examples.FusedLayerNorm.Proofs.FP

/-! The public comparison takes the existing bound on N and a reduction plan. Empty rows have no numerical output; nonempty rows use the existing Welford requirements. -/

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv

open VeriTile.Bench.Examples.FusedLayerNorm.Kernels

open VeriTile Triton FP.Structural FP.CountConversion

open LayerNormFPContract LayerNormFPExecution

open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)

open scoped VeriTile.Spec

open _root_.VeriTile.Triton.FP.Equational (Schedules)

/-- Nonempty rows use exactly Welford's compiled statistics domain. Empty
rows have no numerical output and require no domain checks. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : FP.GuardExpression.Condition :=
  if N = 0 then .top else WelfordFPContract.requirements x N stride empty plans

end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
