import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Kernel
import bench.examples.StableLogSumExp.Kernels
import bench.examples.StableLogSumExp.Proofs.FP

/-! The Real theorem takes positive row size. The FP comparison uses the existing numerical requirements below. -/

noncomputable section

namespace VeriTile.Bench.Examples.StableLogSumExpFPEquiv

open VeriTile.Bench.Examples.StableLogSumExp.Kernels

open VeriTile Triton

open StableLogSumExpFPContract

open scoped VeriTile.Spec

open StableLogSumExpFPExecution

open SoftmaxStableFPContract (rowExpressions center rowPlan)

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

/-- Finite inputs, row maximum, exponentials and sum-tree intermediates;
exp(max) is positive and nonzero, its reciprocal is finite, and the shifted
sum is finite and positive with a finite logarithm. Checks also cover the
zero intermediates used by the scalar derivation. `plans` chooses the shared
addition tree; input values themselves need not be positive. -/
def requirements (x : RegionName) (B : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x B
  FP.LogSumExpShift.checks M xs (center M xs) (rowPlan plans B).tree

end VeriTile.Bench.Examples.StableLogSumExpFPEquiv
