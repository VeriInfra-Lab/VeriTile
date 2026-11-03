import VeriTile.Triton.DSL.InlineAsmWordAdd

/-! A declared integer constexpr may be assigned a runtime value by the source.
Freeze its entry value outside the body, rewrite only reads before the first
top-level assignment, and retain that assignment and every later register read.
Control-flow rebindings and type-changing entry values remain unsupported. -/
namespace VeriTile.Triton.DSL.ReboundEntryConstants
open Lean Elab Term Meta

def entryProfile (profile : CallSignature) (names : List (String × String)) : CallSignature :=
  {profile with constexpr := profile.constexpr.filter (fun n=> !names.any (·.1==n)) ++ names.map Prod.snd}
private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "entry constants must be static names"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "entry constant names must be static"
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def writes (names : List Name) (raw : Syntax) : Bool :=
  let own:=match (⟨raw⟩ : TSyntax `tritonStmt) with
    | `(tritonStmt| $name:ident = $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $name:ident += $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $name:ident -= $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $name:ident *= $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $name:ident /= $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr)=>names.contains name.getId.eraseMacroScopes
    | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*)=>
      (#[a,b]++more).any (fun id=>names.contains id.getId.eraseMacroScopes)
    | `(tritonStmt| for $name:ident in range($_:tritonExpr) { $_:tritonStmt* })=>names.contains name.getId.eraseMacroScopes
    | _=>Bool.false
  let loopBinding:=raw.getNumArgs>1 && raw[0].isAtom && raw[1].isIdent &&
    ["for","tl.for","tl.static_range"].contains raw[0].getAtomVal.trimAscii.toString &&
    names.contains raw[1].getId.eraseMacroScopes
  own || loopBinding || raw.getArgs.any (writes names)
private partial def replaceReads (names : List (Name × Ident)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    match names.find? (·.1==raw.getId.eraseMacroScopes) with
    | some (_,name)=>name.raw | none=>raw
  else raw.setArgs (raw.getArgs.map (replaceReads names))

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete source body required"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let constants ← (← listValues (← mkAppM ``CallSignature.constexpr #[profile])).mapM fun n=>return (← textValue n).toName
  let mutated:=body.toList.filterMap fun stmt=>match stmt with
    | `(tritonStmt| $name:ident = $_:tritonExpr)=>
      if constants.contains name.getId.eraseMacroScopes then some name.getId.eraseMacroScopes else none
    | _=>none
  let mutated:=mutated.eraseDups
  if mutated.isEmpty then return ← InlineAsmWordAdd.elaborateInvocation stx expected
  let used:=ids parsed ++ ids stx ++ constants
  let mut values : List (Name × Ident × Expr):=[]
  for name in mutated do
    let value ← elabTerm (mkIdent name) none
    let ty ← inferType value
    unless (← isDefEq ty (mkConst ``Nat)) || (← isDefEq ty (mkConst ``Int)) do
      throwError "rebound entry constexpr must have an integer carrier"
    let mut fresh:=Name.anonymous
    repeat
      let id ← mkFreshId
      fresh:=Name.mkSimple ("veritile_entry_"++(id.toString.map fun c=>if c.isAlphanum then c else '_'))
      unless used.contains fresh do break
    values:=values++[(name,mkIdent fresh,value)]
  let mut active:=values.map (fun (name,id,_) => (name,id))
  let mut lowered:=#[]
  for stmt in body do
    if let `(tritonStmt| $name:ident = $rhs:tritonExpr):=stmt then
      let rhs : TSyntax `tritonExpr:=⟨replaceReads active rhs⟩
      lowered:=lowered.push (← `(tritonStmt| $name:ident = $rhs:tritonExpr))
      active:=active.filter (·.1 != name.getId.eraseMacroScopes)
    else
      if writes mutated stmt then throwError "entry constexpr rebinding requires a top-level plain assignment"
      lowered:=lowered.push (⟨replaceReads active stmt⟩ : TSyntax `tritonStmt)
  let pairs : List (TSyntax `term) ← values.mapM fun (name,id,_)=>do
    let original : TSyntax `term:=⟨Syntax.mkStrLit name.toString⟩
    let frozen : TSyntax `term:=⟨Syntax.mkStrLit id.getId.toString⟩
    `(($original,$frozen))
  let pairs:=pairs.toArray
  let signature : TSyntax `term:=⟨stx[1]⟩
  let signature ← `(entryProfile $signature [$pairs,*])
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (lowered.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let adjusted:=stx.setArgs ((stx.getArgs.set! 1 signature).set! (stx.getArgs.size-1) raw)
  let rec freeze (todo : List (Name × Ident × Expr)) : TermElabM Expr := do
    match todo with
    | []=>InlineAsmWordAdd.elaborateInvocation adjusted expected
    | (_,id,value)::rest=>
      withLetDecl id.getId (← inferType value) value fun bound=>do
        let result ← freeze rest
        mkLetFVars #[bound] result
  freeze values
end VeriTile.Triton.DSL.ReboundEntryConstants
