import Mathlib.Data.Fin.Rev
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.ReductionSchedule
import VeriTile.Triton.Float.StructuralIO
import bench.examples.RowWiseSum.Kernels
import bench.examples.RowWiseSum.Proofs.FP

/-! The public theorem states the row-size and scheduling conditions. The FP comparison uses the existing reduction domain below. -/

namespace VeriTile.Bench.Examples.RowWiseSumFPEquiv

open VeriTile.Bench.Examples.RowWiseSum.Kernels

open VeriTile Triton

open FP.Structural FP.Equational

open FP.Guarded FP.ReductionSchedule FP.ScalarReduction

set_option maxHeartbeats 1600000

/-- Explicit paths check the intermediate operands used in both tree
normalizations, including padding. They impose no experiment-size restriction. -/
def domain (nCol B : Nat) (plans : Schedules) : FP.GuardExpression.Condition :=
  let p := plan B plans
  let q := reindex p (Fin.revPerm : Equiv.Perm (Fin B))
  FP.GuardExpression.Requirements.all
    (((normalize p.tree).operands ++ (normalize q.tree).operands).map fun t =>
      .guard .finite (value (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)
        (inputExpression nCol B) (FP.ScalarArithmetic.zero
          (FP.GuardExpression.algebra FP.GuardExpression.MemoryInput)) t))

abbrev Rules := FP.ScalarArithmetic.Rules

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.RowWiseSumFPEquiv
