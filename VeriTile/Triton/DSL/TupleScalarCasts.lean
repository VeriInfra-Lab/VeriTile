import VeriTile.Triton.DSL.SignedBooleanAddition

/-! Top-level simultaneous int64 self-casts of distinct integer arguments.
Every right-hand side reads only its own incoming argument, so sequential
lowering preserves the tuple values. Other argument tuples fail closed.
The pasted source remains unchanged; the existing signed-Boolean frontend
and call binder handle the normalized scalar statements. -/
namespace VeriTile.Triton.DSL.TupleScalarCasts
open Lean Elab Term Meta

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "tuple argument types must have literal names"

private def scalarNames (profile : Syntax) : TermElabM (List Name × List Name) := do
  let profile ← elabTermEnsuringType profile (mkConst ``CallSignature)
  let mut allNames := []
  let mut integers := []
  for entry in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry])
      | throwError "scalar argument name must be literal"
    let ty ← whnf (← mkAppM ``Prod.snd #[entry])
    allNames := name.toName :: allNames
    if ty.isConstOf ``ScalarType.integer || ty.isConstOf ``ScalarType.i32 || ty.isConstOf ``ScalarType.i64 then
      integers := name.toName :: integers
  return (allNames,integers)

private def isSelfCast (id : Ident) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| tl.cast($value:ident, tl.int64)) =>
    value.getId.eraseMacroScopes == id.getId.eraseMacroScopes
  | _ =>
    match methodCast? e with
    | some (value, dt) =>
      let wide := match dt with | `(tritonDType| tl.int64) => Bool.true | _ => Bool.false
      match value with
      | `(tritonExpr| $name:ident) => wide && name.getId.eraseMacroScopes == id.getId.eraseMacroScopes
      | _ => Bool.false
    | none => Bool.false

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value => pure value | .error message => throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value => pure value | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete caller body"
  let (allNames,integers) ← scalarNames stx[1]
  let mut lowered := #[]
  for statement in statements do
    match statement with
    | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $x:tritonExpr, $y:tritonExpr $[, $rest:tritonExpr]*) =>
      let names := #[a,b] ++ more
      let values := #[x,y] ++ rest
      let ids := names.toList.map (·.getId.eraseMacroScopes)
      if ids.any allNames.contains then
        unless names.size == values.size && ids.eraseDups.length == ids.length &&
            ids.all integers.contains && (names.zip values).all (fun (name,value) => isSelfCast name value) do
          throwError "caller tuple casts require distinct integer arguments with matching int64 self-casts"
        for (name,value) in names.zip values do
          lowered := lowered.push (← `(tritonStmt| $name:ident = $value:tritonExpr))
      else lowered := lowered.push statement
    | _ => lowered := lowered.push statement
  let raw := Syntax.atom raw.getHeadInfo
    ("{\n"++String.intercalate "\n" (lowered.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  SignedBooleanAddition.elaborateInvocation
    (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.TupleScalarCasts
