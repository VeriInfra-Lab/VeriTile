import VeriTile.Triton.DSL.OptionalPointers

/-! A guarded nullable pointer may be rebased once before its first use.
The source assignment becomes a fresh branch-local SSA pointer. Presence is
still frozen by OptionalPointers, which checks every remaining nullable use.
This optional frontend does not change existing OptionalPointers clients. -/
namespace VeriTile.Triton.DSL.GuardedOptionalRebinding
open Lean Elab Term Meta

syntax (name := withRebasedCall) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " guarded_optional_pointers " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term
syntax (name := withRebasedSignature) "with_triton_signature " term
  " guarded_optional_pointers " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def names (raw : Syntax) : List Name :=
  (if raw.isIdent then [raw.getId.eraseMacroScopes] else []) ++
    raw.getArgs.toList.flatMap names

private partial def rename (old : Name) (new : Ident) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes == old then new.raw
  else raw.setArgs (raw.getArgs.map (rename old new))

private partial def rejectReassignment (pointer : Name) (raw : Syntax) : MacroM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=", "+=", "-=", "*=", "/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString == operator) then
      if (args.take index).any (fun a => a.isIdent && a.getId.eraseMacroScopes == pointer) then
        Macro.throwErrorAt raw "guarded optional pointer permits exactly one rebase"
  if let first :: counter :: _ := args then
    if first.isAtom && ["for", "tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && counter.getId.eraseMacroScopes == pointer then
      Macro.throwErrorAt counter "a loop cannot shadow a guarded optional pointer"
  for arg in raw.getArgs do rejectReassignment pointer arg

private partial def basePointer? (raw : TSyntax `tritonExpr) : Option Name :=
  match raw with
  | `(tritonExpr| $pointer:ident) => some pointer.getId.eraseMacroScopes
  | `(tritonExpr| $left:tritonExpr + $_:tritonExpr) => basePointer? left
  | _ => none

private partial def lower (pointers : List Name) (used : List Name) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonStmt| if $pointer:ident is not None { $yes:tritonStmt* } else { $no:tritonStmt* }) := raw then
    if pointers.contains pointer.getId.eraseMacroScopes then
      let yes ← branch pointer yes
      return (← `(tritonStmt| if $pointer:ident is not None { $yes* } else { $no* })).raw
  if let `(tritonStmt| if $pointer:ident is not None { $yes:tritonStmt* }) := raw then
    if pointers.contains pointer.getId.eraseMacroScopes then
      let yes ← branch pointer yes
      return (← `(tritonStmt| if $pointer:ident is not None { $yes* })).raw
  return raw.setArgs (← raw.getArgs.mapM (lower pointers used))
where
  branch (pointer : Ident) (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt)) := do
    let old := pointer.getId.eraseMacroScopes
    -- Require the rebase to be the first statement in the presence arm.
    let some first := (body[0]? : Option (TSyntax `tritonStmt)) | return body
    let `(tritonStmt| $target:ident = $value:tritonExpr) := first | return body
    unless target.getId.eraseMacroScopes == old && basePointer? value == some old do return body
    if ((names value.raw).filter (· == old)).length != 1 then
      Macro.throwErrorAt value "optional pointer offset cannot read the pointer being rebased"
    let aliasName := Name.mkSimple ("__veritile_optional_" ++ old.toString)
    if used.contains aliasName then Macro.throwErrorAt pointer "guarded optional pointer SSA name collision"
    let fresh := mkIdent aliasName
    let head ← `(tritonStmt| $fresh:ident = $value:tritonExpr)
    let mut result := #[head]
    for statement in body.extract 1 body.size do
      -- Only one assignment to this pointer is supported, and its alias
      -- cannot escape the presence arm. Normal pointer arithmetic may read it.
      rejectReassignment old statement.raw
      result := result.push ⟨rename old fresh statement.raw⟩
    return result

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind == choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out := out ++ " "
    out := out ++ token
    previous := token
  return out

private def elaborate (stx : Syntax) (hasScalars : Bool) (expected : Option Expr) : TermElabM Expr := do
  let bindings := stx[if hasScalars then 8 else 4].getSepArgs
  let pointers ← bindings.toList.mapM fun binding => do
    let `(tritonCallBinding| $pointer:ident := $_:term) := binding
      | throwErrorAt binding "expected an optional pointer binding"
    return pointer.getId.eraseMacroScopes
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}") with
    | .ok parsed => pure parsed
    | .error message => throwErrorAt raw "{message}"
  let lowered ← liftMacroM <| lower pointers (names parsed) parsed
  let `(triton { $statements:tritonStmt* }) := lowered | throwError "expected a complete Triton body"
  let text := "{\n" ++ String.intercalate "\n" (statements.toList.map (fun s => render s.raw)) ++ "\n}"
  let args := stx.getArgs.set! (stx.getArgs.size - 1) (Syntax.atom .none text)
  let kind := if hasScalars then ``OptionalPointers.withCallPresence else ``OptionalPointers.withSignaturePresence
  let delegated := Syntax.node .none kind args
  if hasScalars then OptionalPointers.elaborateInvocation delegated expected
  else OptionalPointers.elaborateSignature delegated expected

@[term_elab withRebasedCall] def elaborateInvocation : TermElab := fun stx expected => elaborate stx Bool.true expected
@[term_elab withRebasedSignature] def elaborateSignature : TermElab := fun stx expected => elaborate stx Bool.false expected
end VeriTile.Triton.DSL.GuardedOptionalRebinding
