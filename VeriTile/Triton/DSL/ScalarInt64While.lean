import VeriTile.Triton.DSL.SignedIntegerAddressAliases

/-! Scalar signed-counter while loops with an invariant bound and a final
int64 unit increment. The loop body, bound cast and final increment are retained.
The internal signed range visits exactly the counter values selected by while;
mutated bounds and additional counter writes are refused. -/
namespace VeriTile.Triton.DSL.ScalarInt64While
open Lean Elab Term

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

private def unit64 (step : TSyntax `tritonExpr) : Bool :=
  match step with
  | `(tritonExpr| tl.full((), 1, tl.int64)) => Bool.true
  | `(tritonExpr| tl.full([], 1, tl.int64)) => Bool.true
  | `(tritonExpr| tl.full([], 1, dtype=tl.int64)) => Bool.true
  | _ => Bool.false

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
  if let `(tritonStmt| while $condition:tritonExpr { $body:tritonStmt* }) := raw then
    let `(tritonExpr| $counter:ident < $bound:tritonExpr) := condition
      | Macro.throwErrorAt condition "int64 while requires a scalar counter < invariant bound"
    let boundName ← match bound with
      | `(tritonExpr| $name:ident) => pure name.getId.eraseMacroScopes
      | `(tritonExpr| tl.cast($name:ident, tl.int64)) => pure name.getId.eraseMacroScopes
      | _ => Macro.throwErrorAt bound "int64 while requires an invariant scalar bound or its int64 cast"
    let some last:=body[body.size-1]? | Macro.throwErrorAt raw "int64 while requires a final unit increment"
    let `(tritonStmt| $updated:ident += $step:tritonExpr) := last
      | Macro.throwErrorAt last "int64 while requires a final int64 unit increment"
    unless updated.getId.eraseMacroScopes==counter.getId.eraseMacroScopes && unit64 step do
      Macro.throwErrorAt last "int64 while requires counter += tl.full((), 1, tl.int64)"
    let inner:=(body.toList.take (body.size-1)).flatMap (fun s=>writes s.raw)
    if inner.contains counter.getId.eraseMacroScopes then
      Macro.throwErrorAt raw "int64 while counter may only be written by the final increment"
    let allWrites:=body.toList.flatMap (fun s=>writes s.raw)
    if allWrites.contains boundName then
      Macro.throwErrorAt raw "int64 while bound must remain unchanged"
    let body ← body.mapM fun s=>do pure (⟨← lower s.raw⟩ : TSyntax `tritonStmt)
    return (← `(tritonStmt| for $counter:ident in range(tl.cast($counter:ident, tl.int64), $bound:tritonExpr, tl.cast(1, tl.int64)) { $body* })).raw
  return raw.setArgs (← raw.getArgs.mapM lower)

private def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.escapeNames source)
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let names ← SignedIntegerAddressAliases.initialKinds profile
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  discard <| liftMacroM (validate names stmts.toList)
  let body ← liftMacroM (lower parsed)
  let `(triton { $stmts:tritonStmt* }) := body | throwError "expected a complete source body"
  let source:="{\n"++String.intercalate "\n" (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))

@[term_elab PointerArguments.withCallOffsets]
def elaborateInvocation : TermElab := fun stx expected => do
  SingleAssignmentPointerOffsets.elaborateInvocation (← SignedIntegerAddressAliases.prepareSource (← prepareSource stx)) expected
@[term_elab PointerArguments.withSignatureOffsets]
def elaborateSignature : TermElab := fun stx expected => do
  SingleAssignmentPointerOffsets.elaborateSignature (← SignedIntegerAddressAliases.prepareSource (← prepareSource stx)) expected
end VeriTile.Triton.DSL.ScalarInt64While
