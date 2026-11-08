import VeriTile.Triton.DSL.FP32CallValues
import VeriTile.Triton.DSL.SignedIntegerAddressAliases

/-! Scalar signed-counter while loops with an invariant bound and a final
immutable natural constexpr increment. The loop body, bound cast and final increment are retained.
For positive strides the internal signed range visits exactly the counter values selected by while;
mutated bounds and additional counter writes are refused. -/
namespace VeriTile.Triton.DSL.ScalarInt64StrideWhile
open Lean Elab Term Meta

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers
private partial def writes (raw : Syntax) : List Name := Id.run do
  let args:=raw.getArgs.toList
  let direct:=match args.findIdx? (fun a=>a.isAtom && ["=",":=","+=","-=","*=","/="].contains a.getAtomVal.trimAscii.toString) with
    | some n => (args.take n).flatMap identifiers
    | none => []
  let binder:=match args with
    | first::counter::_ => if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString && counter.isIdent
        then [counter.getId.eraseMacroScopes] else []
    | _ => []
  return direct++binder++args.flatMap writes

private def stepName? (step : TSyntax `tritonExpr) : Option Name :=
  match step with
  | `(tritonExpr| $name:ident) => some name.getId.eraseMacroScopes
  | _ => none

/-- A range must never turn a floating or unknown source counter into an
integer counter. Require a dominating signed64 source binding on each path. -/
private partial def validate (names : List (Name × Nat))
    (stmts : List (TSyntax `tritonStmt)) : MacroM (List (Name × Nat)) := do
  let mut names:=names
  for stmt in stmts do
    match stmt with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      names:=SignedIntegerAddressAliases.bindName names id.getId.eraseMacroScopes
        (SignedIntegerAddressAliases.kind names value)
    | `(tritonStmt| $id:ident += $value:tritonExpr) =>
      let before:=(names.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd |>.getD 0
      let after:=SignedIntegerAddressAliases.kind names value
      names:=SignedIntegerAddressAliases.bindName names id.getId.eraseMacroScopes
        (if before>0 && after>0 then max before after else 0)
    | `(tritonStmt| while $condition:tritonExpr { $body:tritonStmt* }) =>
      let `(tritonExpr| $counter:ident < $bound:tritonExpr) := condition
        | Macro.throwErrorAt condition "int64 while requires a scalar counter < invariant bound"
      unless (names.find? (·.1==counter.getId.eraseMacroScopes)).map Prod.snd == some 2 do
        Macro.throwErrorAt counter "int64 while counter requires a dominating signed64 binding"
      if let `(tritonExpr| $name:ident) := bound then
        unless ((names.find? (·.1==name.getId.eraseMacroScopes)).map Prod.snd |>.getD 0)>0 do
          Macro.throwErrorAt bound "int64 while bare bound requires known integer type"
      discard <| validate names body.toList
      let changed:=body.toList.flatMap (fun s=>writes s.raw)
      names:=names.map fun p=>if changed.contains p.1 then (p.1,0) else p
    | `(tritonStmt| if $_:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      discard <| validate names yes.toList
      discard <| validate names no.toList
      let changed:=(yes.toList++no.toList).flatMap (fun s=>writes s.raw)
      names:=names.map fun p=>if changed.contains p.1 then (p.1,0) else p
    | `(tritonStmt| if $_:tritonExpr { $body:tritonStmt* }) =>
      discard <| validate names body.toList
      let changed:=body.toList.flatMap (fun s=>writes s.raw)
      names:=names.map fun p=>if changed.contains p.1 then (p.1,0) else p
    | _ =>
      names:=names.map fun p=>if (writes stmt.raw).contains p.1 then (p.1,0) else p
  return names

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  if let `(tritonStmt| $name:ident = $value:num) := raw then
    unless value.getNat < 2^31 do Macro.throwErrorAt value "integer literal assignment requires a representable int32 value"
    return (← `(tritonStmt| $name:ident = tl.cast($value:num, tl.int32))).raw
  if let `(tritonStmt| while $condition:tritonExpr { $body:tritonStmt* }) := raw then
    let `(tritonExpr| $counter:ident < $bound:tritonExpr) := condition
      | Macro.throwErrorAt condition "int64 while requires a scalar counter < invariant bound"
    let boundName ← match bound with
      | `(tritonExpr| $name:ident) => pure name.getId.eraseMacroScopes
      | `(tritonExpr| tl.cast($name:ident, tl.int64)) => pure name.getId.eraseMacroScopes
      | _ => Macro.throwErrorAt bound "int64 while requires an invariant scalar bound or its int64 cast"
    let some last:=body[body.size-1]? | Macro.throwErrorAt raw "int64 while requires a final constexpr increment"
    let `(tritonStmt| $updated:ident += $step:tritonExpr) := last
      | Macro.throwErrorAt last "int64 while requires a final immutable natural constexpr increment"
    unless updated.getId.eraseMacroScopes==counter.getId.eraseMacroScopes && (stepName? step).isSome do
      Macro.throwErrorAt last "int64 stride while requires counter += immutable natural constexpr"
    let inner:=(body.toList.take (body.size-1)).flatMap (fun s=>writes s.raw)
    if inner.contains counter.getId.eraseMacroScopes then
      Macro.throwErrorAt raw "int64 while counter may only be written by the final increment"
    let some strideName := stepName? step | Macro.throwErrorAt step "expected a natural constexpr stride"
    let allWrites:=body.toList.flatMap (fun s=>writes s.raw)
    if allWrites.contains strideName then
      Macro.throwErrorAt raw "int64 while stride must remain unchanged"
    if allWrites.contains boundName then
      Macro.throwErrorAt raw "int64 while bound must remain unchanged"
    let promoted ← `(tritonStmt| $updated:ident += tl.cast($step:tritonExpr, tl.int64))
    let body := body.set! (body.size-1) promoted
    let body ← body.mapM fun s=>do pure (⟨← lower s.raw⟩ : TSyntax `tritonStmt)
    return (← `(tritonStmt| for $counter:ident in range(tl.cast($counter:ident, tl.int64), $bound:tritonExpr, $step:tritonExpr) { $body* })).raw
  return raw.setArgs (← raw.getArgs.mapM lower)

private partial def constexprNames (value : Expr) : TermElabM (List Name) := do
  let value ← Meta.whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "static constexpr names required"
  let .lit (.strVal name) ← Meta.whnf value.getAppArgs[1]! | throwError "literal constexpr name required"
  return Name.mkSimple name :: (← constexprNames value.getAppArgs[2]!)
private partial def checkStrides (constants : List Name) (raw : Syntax) : TermElabM Unit := do
  if let `(tritonStmt| while $_:tritonExpr { $body:tritonStmt* }) := raw then
    let some last := body[body.size-1]? | throwError "empty scalar while"
    let `(tritonStmt| $_:ident += $step:tritonExpr) := last | throwError "final increment required"
    let some name := stepName? step | throwError "natural constexpr stride required"
    unless constants.contains name do throwError "while stride must be a declared natural constexpr"
  for child in raw.getArgs do checkStrides constants child

private def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.escapeNames source)
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let names ← SignedIntegerAddressAliases.initialKinds profile
  let constants ← mkAppM ``Triton.CallSignature.constexpr #[profile]
  let constants ← Meta.whnf constants
  let constants ← constexprNames constants
  checkStrides constants parsed
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  discard <| liftMacroM (validate names stmts.toList)
  let body ← liftMacroM (lower parsed)
  let `(triton { $stmts:tritonStmt* }) := body | throwError "expected a complete source body"
  let source:="{\n"++String.intercalate "\n" (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))

private partial def dedupCasts (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← dedupCasts raw[0]!
  if let `(tritonExpr| tl.cast(tl.cast($value:tritonExpr, tl.int64), tl.int64)) := raw then
    return (← `(tritonExpr| tl.cast($value:tritonExpr, tl.int64))).raw
  return raw.setArgs (← raw.getArgs.mapM dedupCasts)
private def prepareAliases (stx : Syntax) : TermElabM Syntax := do
  let stx ← SignedIntegerAddressAliases.prepareSource stx
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok text => pure text | .error message => throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwError "{message}"
  let `(triton { $body:tritonStmt* }) := (← liftMacroM (dedupCasts parsed)) | throwError "complete body required"
  let source := "{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))

@[term_elab IntegerCallArguments.withIntegers]
def elaborateInvocation : TermElab := fun stx expected => do
  FP32CallValues.elaborateInvocation (← prepareAliases (← prepareSource stx)) expected
end VeriTile.Triton.DSL.ScalarInt64StrideWhile
