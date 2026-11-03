import VeriTile.Triton.DSL.PythonFullConstructors

/-! Optional immutable module-level floating literal bindings. Their metadata
stays outside pasted source. The literal enters the same typed numeric parser
as a literal written directly in the body; no extra statement is emitted. -/
namespace VeriTile.Triton.DSL.ModuleFloatConstants
open Lean Elab Term Meta

def withoutConstants (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with sourceConstants := profile.sourceConstants.filter (fun p=> !names.contains p.1)}

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "module bindings must be statically known"

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "module binding requires a literal string"
  return text

private partial def floatingLiteral (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $_:scientific) => Bool.true
  | `(tritonExpr| -$v:tritonExpr) | `(tritonExpr| ($v:tritonExpr)) => floatingLiteral v
  | _ => Bool.false

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i:=args.findIdx? (fun x=>x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x=>(ids x).any names.contains) then
        throwError "module floating constants cannot be reassigned"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then throwError "a loop cannot shadow a module constant"
  for child in raw.getArgs do rejectWrites names child

private partial def substitute (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    match values.find? (fun p=>p.1==raw.getId.eraseMacroScopes) with
    | some (_,value)=>value.raw
    | none=>raw
  else if let `(tritonExpr| $id:ident):=raw then
    match values.find? (fun p=>p.1==id.getId.eraseMacroScopes) with
    | some (_,value)=>value.raw
    | none=>raw
  else raw.setArgs (raw.getArgs.map (substitute values))

private partial def tokens : Syntax → List String
  | .atom _ value=>[value]
  | .ident _ spelling _ _=>[spelling.toString]
  | .node _ k args=>if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing=>[]
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

/-- Validate original namespaces before removing the bindings already lowered
by this optional adapter. Remaining integer/module bindings use the base path. -/
def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let constants ← listValues (← mkAppM ``CallSignature.sourceConstants #[profile])
  let environment ← getEnv
  let mut values : List (Name × TSyntax `tritonExpr):=[]
  let mut moduleNames : List String:=[]
  for entry in constants do
    let name ← stringValue (← mkAppM ``Prod.fst #[entry])
    let source ← stringValue (← mkAppM ``Prod.snd #[entry])
    moduleNames:=moduleNames++[name]
    let parsed : TSyntax `tritonExpr ← match Parser.runParserCategory environment `tritonExpr source with
      | .ok parsed=>pure ⟨parsed⟩
      | .error _=>throwError "source constant must be a numeric literal"
    if floatingLiteral parsed then
      discard <| liftMacroM <| SourceBindings.constant environment name "0"
      values:=values++[(name.toName,parsed)]
  if values.isEmpty then return stx
  let mut parameters : List String:=[]
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for entry in ← listValues (← mkAppM field #[profile]) do
      parameters:=parameters++[← stringValue (← mkAppM ``Prod.fst #[entry])]
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    parameters:=parameters++[← stringValue entry]
  for entry in ← listValues (← mkAppM ``CallSignature.unaryImports #[profile]) do
    moduleNames:=moduleNames++[← stringValue (← mkAppM ``Prod.fst #[entry])]
  unless moduleNames.eraseDups.length==moduleNames.length && moduleNames.all (fun n=> !parameters.contains n) do
    throwError "module constants must be distinct from parameters and each other"
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory environment `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwError "{message}"
  rejectWrites (values.map Prod.fst) parsed
  let lowered:=substitute values parsed
  let `(triton { $stmts:tritonStmt* }):=lowered | throwError "expected a complete body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (stmts.toList.map (render ·.raw))++"\n}")
  let names : Array (TSyntax `term) := (values.map (fun p=>⟨Syntax.mkStrLit p.1.toString⟩)).toArray
  let signature : TSyntax `term:=⟨stx[1]⟩
  let signature ← `(withoutConstants $signature [$names,*])
  return stx.setArgs ((stx.getArgs.set! 1 signature).set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  PythonFullConstructors.elaborateInvocation (← prepareSource stx) expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected=>do
  PythonFullConstructors.elaborateSignature (← prepareSource stx) expected
end VeriTile.Triton.DSL.ModuleFloatConstants
