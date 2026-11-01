import VeriTile.Triton.DSL.PythonInlineSuites

/-! Accept source `tl.static_range` bounds made from natural constexpr values.
The optional spelling adapter keeps the complete source quotation, checks every
bound identifier against the call signature, and lowers to the existing static
loop. Aligned arange endpoints use the checked partition-width identity below. -/
namespace VeriTile.Triton.DSL.StaticRanges
open Lean Elab Term Meta

scoped syntax (name := staticRange) (priority := low)
  "for " ident " in " "tl.static_range(" tritonExpr ")" " { " tritonStmt* " }" : tritonStmt

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then
    return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "constexpr names must form a static list"

private partial def natural (names : List Name) (raw : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match raw with
  | `(tritonExpr| $n:num) => `(($n:num : Nat))
  | `(tritonExpr| $id:ident) =>
    unless names.contains id.getId.eraseMacroScopes do
      Macro.throwErrorAt id "static_range bounds require declared natural constexprs"
    `(($id:ident : Nat))
  | `(tritonExpr| ($x:tritonExpr)) => natural names x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
    let a ← natural names a; let b ← natural names b; `($a + $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
    let a ← natural names a; let b ← natural names b; `($a * $b)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
    let a ← natural names a; let b ← natural names b; `($a / $b)
  | _ => Macro.throwErrorAt raw "unsupported natural constexpr expression in static_range"

private partial def rejectMutation (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      for id in args.take index do
        if id.isIdent && names.contains id.getId.eraseMacroScopes then
          throwErrorAt id "static_range constexprs must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "static_range cannot shadow a constexpr"
  for arg in raw.getArgs do rejectMutation names arg

private partial def checkCounterBindings (raw : Syntax) : TermElabM Unit := do
  if raw.isOfKind ``staticRange then
    for inner in raw[7].getArgs do
      rejectMutation [raw[1].getId.eraseMacroScopes] inner
  for arg in raw.getArgs do checkCounterBindings arg

private partial def rewrite (names counters : List Name) (raw : Syntax) : MacroM Syntax := do
  if raw.isOfKind ``staticRange then
    let id : Ident := ⟨raw[1]⟩
    let n ← natural names ⟨raw[4]⟩
    let body : Array (TSyntax `tritonStmt) ← raw[7].getArgs.mapM fun s =>
      return ⟨← rewrite names (id.getId.eraseMacroScopes::counters) s⟩
    return (← `(tritonStmt| for $id:ident in range($($n)) { $body* })).raw
  let raw := raw.setArgs (← raw.getArgs.mapM (rewrite names counters))
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.arange($i:ident * $w:ident, ($j:ident + $one:num) * $v:ident)) =>
    if i.getId.eraseMacroScopes == j.getId.eraseMacroScopes &&
        w.getId.eraseMacroScopes == v.getId.eraseMacroScopes && one.getNat == 1 &&
        names.contains w.getId.eraseMacroScopes && counters.contains i.getId.eraseMacroScopes then
      return (← `(tritonExpr| $i:ident * $w:ident + tl.arange(0, $w:ident))).raw
    Macro.throwErrorAt raw "aligned arange requires the active static counter and an unchanged constexpr width"
  | _ => return raw

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args =>
    if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && !token.startsWith "." && token != "(" && token != "[" && previous != "." then out := out++" "
    out := out++token
    previous := token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut constants : List Name := []
  for item in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let item ← whnf item
    let .lit (.strVal name) := item | throwError "constexpr names must be literal"
    constants := constants ++ [Name.mkSimple name]
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (Parser.parserExtension.activateScoped (← getEnv)
      `VeriTile.Triton.DSL.StaticRanges) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  for stmt in stmts do
    rejectMutation constants stmt.raw
    checkCounterBindings stmt.raw
  let stmts ← liftMacroM (stmts.mapM fun s => rewrite constants [] s.raw)
  let source := String.intercalate "\n" (stmts.toList.map render)
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  CallSignature.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected

/-- The aligned endpoint expression has exactly one partition's width. -/
theorem aligned_width (split width : Nat) : (split+1)*width-split*width=width := by
  rw [Nat.add_mul,Nat.one_mul,Nat.add_sub_cancel_left]

end VeriTile.Triton.DSL.StaticRanges
