import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Kernel
import bench.examples.OnlineSoftmax.Kernels
import bench.examples.OnlineSoftmax.Proofs.FP

/-! The public numerical statements require positive logical row length. The row domain below records the definedness conditions for the online and batch operations. -/

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv

open VeriTile.Bench.Examples.OnlineSoftmax.Kernels

open VeriTile Triton FP.Exponential OnlineSoftmaxFPContract

open scoped VeriTile.Spec

open _root_.VeriTile.Triton.FP.Equational (Schedules)

open FP.GuardExpression

open SoftmaxStableFPContract (rowPlan center)

open OnlineSoftmaxFPComparison (rowExpressions)

/-- Check every online update and both normalization paths, including finite
sum intermediates and nonzero divisors. Also check the intermediate operands
needed to relate the sequential sum to the batch reduction tree. -/
def requirements (x : RegionName) (N : Nat) (plans : Schedules) : Condition :=
  let M := FP.GuardExpression.algebra MemoryInput
  let xs := rowExpressions x N
  let row := fun i : Fin N => xs i.val
  .all [FP.OnlineSoftmaxConditions.iterations M xs N,
    FP.SoftmaxShift.checks M row (FP.OnlineSoftmax.state M xs N).1 (prefixPlan N).tree,
    .guard .finite (FP.OnlineSoftmax.state M xs N).2,
    FP.SoftmaxShift.checks M row (center M row) (rowPlan plans N).tree,
    FP.WelfordConditions.schedule M (FP.SoftmaxShift.exponentials M row) (prefixPlan N) (rowPlan plans N)]

end VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv
