import VeriTile.Triton.DSL.OrderedRowSum
open Lean Elab Command VeriTile.Triton.DSL

/-! Invocation metadata must not silently select a different source reduction
or add numerical laws. Reject incompatible axes, options and missing operands. -/
run_cmd liftTermElabM do
  liftMacroM do
    let tree ← `(term| selectedTree)
    let valid ← `(tritonStmt| total = tl.sum(values, 1))
    let wrongAxis ← `(tritonStmt| total = tl.sum(values, 0))
    let extraOption ← `(tritonStmt| total = tl.sum(values, axis=1, keep_dims=true))
    let wrongOperand ← `(tritonStmt| total = tl.sum(other, 1))
    for (bindings,stmt) in [
        ([("values",tree)],wrongAxis),
        ([("values",tree)],extraOption),
        ([("values",tree)],wrongOperand),
        ([("values",tree),("values",tree)],valid)] do
      let rejected ← try
        let _ ← OrderedRowSum.lower bindings [stmt]
        pure Bool.false
      catch _ => pure Bool.true
      unless rejected do Macro.throwError "invalid ordered row sum metadata was accepted"
    let result ← OrderedRowSum.lower [("values",tree)] [valid]
    unless result.length == 1 do Macro.throwError "valid ordered row sum was lost"
