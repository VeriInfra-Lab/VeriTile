import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import bench.examples.Welford.Kernels
import bench.examples.Welford.Proofs.FP

/-! The public entries state their row-size and reduction-plan conditions. The FP domain below records the existing Welford operation requirements. -/

namespace VeriTile.Bench.Examples.WelfordFPEquiv

open VeriTile.Bench.Examples.Welford.Kernels

open VeriTile Triton FP.Structural FP.CountConversion

open WelfordFPContract WelfordFPExecution

open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)

open scoped VeriTile.Spec

open WelfordFPComparison (rowPlan)

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

/-- The numerical proof domain contains initialization, each iteration, and
both normalization paths for all three statistics. Dimensions stay symbolic. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : Condition :=
  FP.WelfordConditions.complete (FP.GuardExpression.algebra MemoryInput)
    (rowExpressions x stride) empty (rowPlan plans N)

end VeriTile.Bench.Examples.WelfordFPEquiv
