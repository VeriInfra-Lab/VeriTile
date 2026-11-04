import VeriTile.Triton.DSL.ConditionalPureJitFunctions

/-! Implicit Boolean-to-int64 promotion in source integer additions. A known
signed64 operand fixes the result channel; comparisons contribute exact 0/1.
Other Boolean additions fail closed. Control-flow assignments invalidate their
outer type facts. The pasted source, casts and statement order stay unchanged. -/
namespace VeriTile.Triton.DSL.IntegralBooleanOperands
open Lean Elab Term Meta

private inductive Kind where
  | signed64 | integer | boolean | unknown
  deriving BEq, Inhabited
private abbrev Types := List (Name × Kind)

private def numeric (a b : Kind) : Kind :=
  if a==.signed64 && (b==.signed64 || b==.integer) ||
      b==.signed64 && a==.integer then .signed64
  else if a==.integer && b==.integer then .integer else .unknown

private partial def kind (types : Types) (value : TSyntax `tritonExpr) : Kind :=
  match value with
  | `(tritonExpr| $id:ident) => (types.lookup id.getId.eraseMacroScopes).getD .unknown
  | `(tritonExpr| $_:num) => .integer
  | `(tritonExpr| ($x:tritonExpr)) | `(tritonExpr| -$x:tritonExpr) => kind types x
  | `(tritonExpr| tl.cast($_:tritonExpr, tl.int64)) => .signed64
  | `(tritonExpr| true) | `(tritonExpr| false) => .boolean
  | `(tritonExpr| $_:tritonExpr < $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr <= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr > $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr >= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr == $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr != $_:tritonExpr) => .boolean
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => numeric (kind types a) (kind types b)
  | _ => .unknown

private partial def lower (types : Types) (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (lower types))
  if let `(tritonExpr| $a:tritonExpr + $b:tritonExpr) := raw then
    let ka:=kind types a
    let kb:=kind types b
    if kb==.boolean then
      unless ka==.signed64 do
        Macro.throwErrorAt raw "Boolean addition requires a known int64 operand"
      return (← `(tritonExpr| $a:tritonExpr + tl.cast($b:tritonExpr, tl.int64))).raw
    if ka==.boolean then
      unless kb==.signed64 do
        Macro.throwErrorAt raw "Boolean addition requires a known int64 operand"
      return (← `(tritonExpr| tl.cast($a:tritonExpr, tl.int64) + $b:tritonExpr)).raw
  return raw

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "Boolean addition requires static caller parameter names"

private def initialTypes (raw : Syntax) : TermElabM Types := do
  let profile ← elabTermEnsuringType raw (mkConst ``CallSignature)
  let mut types : Types:=[]
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair])
      | throwError "scalar name must be literal"
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    let k:=if dtype.isAppOf ``ScalarType.i64 then Kind.signed64
      else if dtype.isAppOf ``ScalarType.integer || dtype.isAppOf ``ScalarType.i32 then .integer
      else .unknown
    types:=(name.toName,k)::types
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf name | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    let dtype ← inferType value
    let k ← if ← isDefEq dtype (mkConst ``Nat) then pure Kind.integer
      else if ← isDefEq dtype (mkConst ``Bool) then pure Kind.boolean
      else pure Kind.unknown
    types:=(name.toName,k)::types
  return types

private partial def writtenNames (raw : Syntax) : List Name :=
  let args:=raw.getArgs
  if args.size≥2 && args[0]!.isAtom && args[0]!.getAtomVal.trimAscii.toString=="for" && args[1]!.isIdent then
    args[1]!.getId.eraseMacroScopes::args.toList.flatMap writtenNames
  else if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* =
      $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) := raw then
    (#[a,b]++more).toList.map (·.getId.eraseMacroScopes)
  else if let `(tritonStmt| $id:ident = $_:tritonExpr) := raw then [id.getId.eraseMacroScopes]
  else if let `(tritonStmt| $id:ident += $_:tritonExpr) := raw then [id.getId.eraseMacroScopes]
  else if let `(tritonStmt| $id:ident -= $_:tritonExpr) := raw then [id.getId.eraseMacroScopes]
  else if let `(tritonStmt| for $id:ident in range($_:tritonExpr, $_:tritonExpr, $_:tritonExpr) { $body:tritonStmt* }) := raw then
    id.getId.eraseMacroScopes::body.toList.flatMap (fun s=>writtenNames s.raw)
  else if let `(tritonStmt| for $id:ident in range($_:tritonExpr, $_:tritonExpr) { $body:tritonStmt* }) := raw then
    id.getId.eraseMacroScopes::body.toList.flatMap (fun s=>writtenNames s.raw)
  else raw.getArgs.toList.flatMap writtenNames

/-- Prepare exact Boolean-to-int64 operand promotion before range lowering. -/
def prepare (stx : Syntax) : TermElabM Syntax := do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete caller body"
  let mut types ← initialTypes stx[1]
  let mut result : Array (TSyntax `tritonStmt):=#[]
  for statement in statements do
    let plainAssignment := match statement with
      | `(tritonStmt| $_:ident = $_:tritonExpr) => Bool.true
      | _ => Bool.false
    let invalidated := if plainAssignment then [] else writtenNames statement.raw
    let localTypes := types.filter (fun entry=>!invalidated.contains entry.1)
    let lowered ← liftMacroM <| lower localTypes statement.raw
    result:=result.push ⟨lowered⟩
    if let `(tritonStmt| $id:ident = $rhs:tritonExpr) := lowered then
      let name:=id.getId.eraseMacroScopes
      types:=(name,kind localTypes rhs)::localTypes.filter (fun entry=>entry.1!=name)
    else
      types:=localTypes
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (result.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let rewritten:=stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
  return rewritten

end VeriTile.Triton.DSL.IntegralBooleanOperands
