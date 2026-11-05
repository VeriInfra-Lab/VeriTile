import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.Kernel
import bench.examples.SoftmaxStable.Kernels
import bench.examples.SoftmaxStable.Proofs.FP

/-! The Real theorem takes positive row size. The FP comparison uses the existing exponential and reduction requirements below. -/

namespace VeriTile.Bench.Examples.SoftmaxStableFPEquiv

open VeriTile.Bench.Examples.SoftmaxStable.Kernels

open VeriTile Triton FP.Structural FP.Exponential

open SoftmaxStableFPContract SoftmaxStableFPExecution

open scoped VeriTile.Spec

open SoftmaxStableFPContract (rowExpressions center rowPlan)

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

/-- Check finite inputs, the row maximum, exponentials and sum intermediates,
plus the nonzero denominators needed by the scalar derivation. The same
addition tree is used by both kernels; inputs need not be positive. -/
def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x B
  FP.SoftmaxShift.checks M xs (center M xs) (rowPlan plans B).tree

end VeriTile.Bench.Examples.SoftmaxStableFPEquiv
