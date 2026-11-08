import VeriTile.Triton.DSL.SignedCallRanges

/-! Propagate a chain of sole nested conditional returns to the enclosing
function continuation. A scalar lazy conditional preserves short-circuit
execution of each guard. The pasted source is unchanged. -/
namespace VeriTile.Triton.DSL.NestedReturns
open Lean Elab Term
private partial def guard? (statement : TSyntax `tritonStmt) : MacroM (Option (TSyntax `tritonExpr)) := do
  match statement with
  | `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
    match body.toList with
    | [inner] =>
      if inner.raw.getKind==``tritonReturn then
        return some (← `(tritonExpr| (true if $condition:tritonExpr else false)))
      if let some nested ← guard? inner then
        return some (← `(tritonExpr| ($nested:tritonExpr if $condition:tritonExpr else false)))
      return none
    | _ => return none
  | _ => return none
private partial def containsReturn (raw : Syntax) : Bool :=
  raw.getKind==``tritonReturn || raw.getArgs.any containsReturn

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  if let some condition ← guard? ⟨raw⟩ then
    return (← `(tritonStmt| if $condition:tritonExpr { return })).raw
  let first := if raw.getArgs.isEmpty then "" else raw[0]!.getAtomVal
  if containsReturn raw && (raw.getKind==``tritonReturn ||
      ["if","tl.if","for","tl.for","while","tl.static_range"].contains first) then
    Macro.throwErrorAt raw "unsupported conditional return: expected a chain of sole conditional statements"
  return raw.setArgs (← raw.getArgs.mapM lower)
@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwError "{message}"
  let `(triton { $body:tritonStmt* }) := (← liftMacroM (lower parsed)) | throwError "complete body required"
  let source := "{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  let stx := stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))
  SignedCallRanges.elaborateInvocation stx expected
end VeriTile.Triton.DSL.NestedReturns
