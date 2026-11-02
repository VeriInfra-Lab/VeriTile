import VeriTile.Triton.DSL.PythonFullConstructors

/-! Preserve Python's conditional signed pointer advance spelling. The checked
subset selects subtraction or addition of one immutable natural constexpr;
its Boolean selector is also immutable caller metadata. The typed core checks
the target and retains its existing address projection. No floating operation
or control condition is removed. -/
namespace VeriTile.Triton.DSL.ConditionalPointerAdvances
open Lean Elab Term Meta

syntax (name := conditionalAdvance) ident "+=" "-" ident "if" ident "else" ident : tritonStmt

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifiers
private partial def listNames (value : Expr) : TermElabM (List Name) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    let .lit (.strVal name) ← whnf value.getAppArgs[1]! | throwError "constexpr names must be literal"
    return name.toName :: (← listNames value.getAppArgs[2]!)
  throwError "constexpr names must be statically known"

private partial def immutable (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let reject (name : Ident) : TermElabM Unit := do
    if names.contains name.getId.eraseMacroScopes then
      throwError "conditional advance metadata must remain immutable"
  match raw with
  | `(tritonStmt| $name:ident = $_:tritonExpr)
  | `(tritonStmt| $name:ident += $_:tritonExpr)
  | `(tritonStmt| $name:ident -= $_:tritonExpr)
  | `(tritonStmt| $name:ident *= $_:tritonExpr)
  | `(tritonStmt| $name:ident /= $_:tritonExpr)
  | `(tritonStmt| $name:ident := $_:tritonExpr)
  | `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr) => reject name
  | `(tritonStmt| $name:ident += -$_:ident if $_:ident else $_:ident) => reject name
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
    for name in #[a,b]++more do reject name
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = ($_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*)) =>
    for name in #[a,b]++more do reject name
  | _ => pure ()
  if let a::b::_ := raw.getArgs.toList then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        (identifiers b).any names.contains then
      throwError "loop shadows conditional advance metadata"
  for child in raw.getArgs do immutable names child

private partial def lower (constants : List Name) (whole raw : Syntax) : TermElabM Syntax := do
  if raw.getKind == choiceKind then return ← lower constants whole raw[0]!
  if let `(tritonStmt| $pointer:ident += -$negative:ident if $flag:ident else $positive:ident) := raw then
    unless negative.getId.eraseMacroScopes==positive.getId.eraseMacroScopes do
      throwError "conditional pointer advances require the same positive and negative pitch"
    for name in [negative,flag] do
      unless constants.contains name.getId.eraseMacroScopes do
        throwError "conditional pointer advances require caller constexpr pitch and selector"
    let pitch ← elabTerm negative none
    unless ← isDefEq (← inferType pitch) (mkConst ``Nat) do
      throwError "conditional pointer pitch must have a Nat carrier"
    let selector ← elabTerm flag none
    unless ← isDefEq (← inferType selector) (mkConst ``Bool) do
      throwError "conditional pointer selector must be Boolean"
    immutable [negative.getId.eraseMacroScopes,flag.getId.eraseMacroScopes] whole
    return (← `(tritonStmt| if $flag:ident {
      $pointer:ident -= $negative:ident
    } else {
      $pointer:ident += $positive:ident
    })).raw
  return raw.setArgs (← raw.getArgs.mapM (lower constants whole))

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let prepared ← PythonFullConstructors.prepareSource stx
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let constants ← listNames (← mkAppM ``CallSignature.constexpr #[profile])
  let raw := prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value => pure value | .error message => throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value => pure value | .error message => throwError "{message}"
  let body ← lower constants parsed parsed
  let `(triton { $statements:tritonStmt* }) := body | throwError "expected a complete source body"
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let prepared := prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)
  CallSignature.elaborateInvocation prepared expected

end VeriTile.Triton.DSL.ConditionalPointerAdvances
