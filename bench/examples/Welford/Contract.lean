import VeriTile.Triton.Float.WelfordConditions

/-! Welford's FP input domain uses the actual row addresses and FP32 sum tree. -/

namespace VeriTile.Bench.Examples
open VeriTile Triton
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan Schedules)

namespace WelfordFPComparison
def rowPlan (plans : Schedules) (N : Nat) : ReductionPlan N :=
  plans (some .fp32) [N] ⟨0, by simp⟩ Bool.false PUnit.unit
end WelfordFPComparison

namespace WelfordFPContract
open FP.GuardExpression
open WelfordFPComparison (rowPlan)

def rowExpressions (x : RegionName) (stride : Nat) (i : Nat) : Expr MemoryInput :=
  .fp32Load (.input ⟨x, fun pid => pid * stride + i, .real⟩)

/-- The numerical proof domain contains initialization, each iteration, and
both normalization paths for all three statistics. Dimensions stay symbolic. -/
def requirements (x : RegionName) (N stride : Nat) (empty : ReductionPlan 0)
    (plans : Schedules) : Condition :=
  FP.WelfordConditions.complete (FP.GuardExpression.algebra MemoryInput)
    (rowExpressions x stride) empty (rowPlan plans N)

end WelfordFPContract

namespace WelfordFPEquiv
abbrev requirements := WelfordFPContract.requirements
end WelfordFPEquiv

end VeriTile.Bench.Examples
