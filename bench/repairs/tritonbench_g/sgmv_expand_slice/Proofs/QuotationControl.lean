import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.QuotationControl
open VeriTile.Triton FP.Structural
noncomputable section

/-- The control-flow rule used before branch-local source typing. This is not
an equivalence with the old quotation's incorrectly typed mixed-output store. -/
theorem distribute_continuation {α : Type} [Inhabited α] (M : Algebra α)
    (condition : Op .bool []) (yes tail : List ComputeStmt) (s : State α) :
    run M (.ifThen condition yes :: tail) s =
      run M [.ifThenElse condition (yes++tail) tail] s := by
  cases hc : evalOp M none condition s with
  | none => simp [run,step,hc]
  | some value =>
      cases hb : value PUnit.unit <;>
        simp [run,step,hc,hb,run_append]

#axiomsClean distribute_continuation
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.QuotationControl
