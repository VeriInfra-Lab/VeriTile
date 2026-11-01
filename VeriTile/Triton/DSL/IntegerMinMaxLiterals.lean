import VeriTile.Triton.DSL.PythonInlineSuites

/-! Retain weak zero literals in min/max on known signed metadata.
The base floating min/max path is unchanged. This optional pass infers only
straight-line integer expressions from declared scalar/constexpr parameters,
integer loads and program indices; it does not specialize runtime values. -/
namespace VeriTile.Triton.DSL.IntegerMinMaxLiterals
open Lean Elab Term Meta

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then
    return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "source profile must have static names"
private def stringValue (e : Expr) : TermElabM String := do
  let .lit (.strVal s) ← whnf e | throwError "source name must be literal"
  return s

private partial def pointerBase (e : TSyntax `tritonExpr) : Option Name :=
  match e with
  | `(tritonExpr| $id:ident) => some id.getId.eraseMacroScopes
  | `(tritonExpr| ($e:tritonExpr)) => pointerBase e
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr) => pointerBase base
  | _ => none

/-- 0: unknown/noninteger, 1: natural/integer literal, 2: signed metadata. -/
private partial def kind (names : List (Name × Nat)) (pointers : List Name)
    (e : TSyntax `tritonExpr) : Nat :=
  match e with
  | `(tritonExpr| $id:ident) => (names.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd |>.getD 0
  | `(tritonExpr| $n:num) => if n.getNat<2^63 then 1 else 0
  | `(tritonExpr| ($e:tritonExpr)) => kind names pointers e
  | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) =>
    if (pointerBase p).any pointers.contains then 2 else 0
  | `(tritonExpr| tl.program_id($_:tritonExpr)) | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr)) => 1
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr)
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr))
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
    let ka := kind names pointers a
    let kb := kind names pointers b
    if ka==0 || kb==0 then 0 else max ka kb
  | _ => 0

private partial def rewrite (names : List (Name × Nat)) (pointers : List Name) (raw : Syntax) : MacroM Syntax := do
  let selected := kind names pointers ⟨raw⟩ == 2
  let raw := raw.setArgs (← raw.getArgs.mapM (rewrite names pointers))
  if selected then
    match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| tl.maximum($a:tritonExpr, $n:num)) =>
      unless n.getNat==0 do return raw
      return (← `(tritonExpr| tl.maximum($a, tl.cast($n:num, tl.int64)))).raw
    | `(tritonExpr| tl.minimum($a:tritonExpr, $n:num)) =>
      unless n.getNat==0 do return raw
      return (← `(tritonExpr| tl.minimum($a, tl.cast($n:num, tl.int64)))).raw
    | _ => return raw
  return raw

private partial def writes (raw : Syntax) : List Name := Id.run do
  let args := raw.getArgs.toList
  let mut names := args.flatMap writes
  for op in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun s => s.isAtom && s.getAtomVal.trimAscii.toString==op) then
      names:=names++((args.take i).filterMap fun s => if s.isIdent then some s.getId.eraseMacroScopes else none)
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString && b.isIdent then
      names:=b.getId.eraseMacroScopes::names
  return names
private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ k args => if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && !token.startsWith "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut pointers : List Name := []
  for pair in ← listValues (← mkAppM ``Triton.CallSignature.pointers #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    if [``Triton.SourceDType.i32,``Triton.SourceDType.i64].any dtype.isConstOf then
      pointers:=pointers++[(← stringValue (← mkAppM ``Prod.fst #[pair])).toName]
  let mut names : List (Name × Nat) := []
  for pair in ← listValues (← mkAppM ``Triton.CallSignature.scalars #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    if [``Triton.ScalarType.integer,``Triton.ScalarType.i32,``Triton.ScalarType.i64].any dtype.isConstOf then
      names:=names++[((← stringValue (← mkAppM ``Prod.fst #[pair])).toName,1)]
  for name in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let name := (← stringValue name).toName
    let value ← elabTerm (mkIdent name) none
    let type ← whnf (← inferType value)
    if type.isConstOf ``Nat then names:=names++[(name,1)]
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  let mut lowered : Array Syntax := #[]
  for stmt in stmts do
    let loweredStmt ← match stmt with
      | `(tritonStmt| $_:ident = $_:tritonExpr) => liftMacroM (rewrite names pointers stmt.raw)
      | _ => pure stmt.raw
    lowered:=lowered.push loweredStmt
    let assigned := writes stmt.raw
    let valueKind := match stmt with
      | `(tritonStmt| $_:ident = $value:tritonExpr) => kind names pointers value
      | _ => 0
    names:=names.filter (fun p => !assigned.contains p.1)
    if let `(tritonStmt| $id:ident = $_:tritonExpr) := stmt then
      if valueKind>0 then names:=(id.getId.eraseMacroScopes,valueKind)::names
  let source := String.intercalate "\n" (lowered.toList.map render)
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  CallSignature.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.IntegerMinMaxLiterals
