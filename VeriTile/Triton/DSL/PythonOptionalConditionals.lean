import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.PythonConditionGrouping
import VeriTile.Triton.DSL.OptionalPointers

/-! Python assignment conditionals keep their original source spelling. A
nullable-pointer conditional is lowered to its two guarded assignment branches
before the existing presence checker. The untaken load is not evaluated. -/
namespace VeriTile.Triton.DSL.PythonOptionalConditionals
open Lean Elab Term Meta

private partial def guardedAssignments (booleans : List Name) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← guardedAssignments booleans raw[0]!
  if let `(tritonStmt| if $a:ident and $b:ident { $body:tritonStmt* }) := raw then
    if booleans.contains a.getId.eraseMacroScopes && booleans.contains b.getId.eraseMacroScopes then
      let nested ← `(tritonStmt| if $a:ident { if $b:ident { $body* } })
      return ← guardedAssignments booleans nested.raw
  if let `(tritonStmt| $name:ident = ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr)) := raw then
    let pointerGuard := match condition with
      | `(tritonExpr| $_:ident is not None) | `(tritonExpr| $_:ident is None) => Bool.true
      | _ => Bool.false
    if pointerGuard then
      return (← `(tritonStmt| if $condition:tritonExpr {
        $name:ident = $yes:tritonExpr
      } else {
        $name:ident = $no:tritonExpr
      })).raw
  return raw.setArgs (← raw.getArgs.mapM (guardedAssignments booleans))

private partial def stringList (value : Expr) : TermElabM (List String) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    let .lit (.strVal head) ← whnf value.getAppArgs[1]! | throwError "constexpr name must be literal"
    return head :: (← stringList value.getAppArgs[2]!)
  throwError "constexpr names must be statically known"

private def booleanConstants (stx : Syntax) : TermElabM (List Name) := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let names ← stringList (← mkAppM ``CallSignature.constexpr #[profile])
  let mut selected := []
  for name in names do
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Bool) then selected := selected ++ [name.toName]
  return selected

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let booleans ← booleanConstants stx
  let stx ← PythonConditionGrouping.prepareSource stx
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let body ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>liftMacroM (guardedAssignments booleans value)
    | .error message=>throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := body | throwError "expected a complete conditional body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  CallSignature.elaborateInvocation (← prepare stx) expected

@[term_elab OptionalPointers.withCallPresence]
def elaboratePresence : TermElab := fun stx expected => do
  OptionalPointers.elaborateInvocation (← prepare stx) expected
end VeriTile.Triton.DSL.PythonOptionalConditionals
