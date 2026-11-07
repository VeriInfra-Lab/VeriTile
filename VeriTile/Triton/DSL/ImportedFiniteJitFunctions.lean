import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.PythonInlineSuites

/-! Module imports and copied unary dtype-dispatch helpers supplied outside a
copied caller body. The existing finite-helper checker validates the complete
return expression and its exact extern aliases. This adds no runtime statement
and reuses the existing pointer, dtype, load and store lowering unchanged. -/
namespace VeriTile.Triton

structure ImportedFiniteJitFunction where
  parameter : String
  source : String
  imports : List (String × String)

namespace DSL.ImportedFiniteJitFunctions
open Lean Elab Term Meta

syntax (name := importedFunction) "with_triton_imports " "[" tritonCallBinding,* "]"
  " in " "triton_function " "(" ident ")" RawSourceText.body : term
syntax (name := withFunctions) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " signed_pointer_offsets " "[" tritonCallBinding,* "]"
  " pure_functions " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private def sourceText (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok source => pure source
  | .error message => throwErrorAt raw "{message}"

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "helper source and imports must be static strings"
  return text

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "helper imports must be statically known"

private def declarations (helper parameter : Ident) (source : String)
    (imports : List (String × String)) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let source := source.trimAscii.toString
  unless source.startsWith "return " && !source.contains '$' do
    throwError "imported finite helper requires one copied return expression"
  let expression ← match Parser.runParserCategory (← getEnv) `tritonExpr
      ("(" ++ (source.drop 7).toString ++ ")") with
    | .ok parsed => pure (⟨parsed⟩ : TSyntax `tritonExpr)
    | .error _ => throwError "imported finite helper requires one copied return expression"
  let mut code := #[]
  for (aliasName,target) in imports do
    let aliasName := mkIdent aliasName.toName
    let target := mkIdent target.toName
    code := code.push (← `(tritonStmt| tl.source_alias $aliasName:ident = $target:ident))
  code := code.push (← `(tritonStmt| tl.source_helper $helper:ident($parameter:ident) { return $expression:tritonExpr }))
  -- Check the actual complete source before it can be bound by a caller.
  let _ ← liftMacroM <| FiniteHelperSource.prepare code.toList
  return code

@[term_elab importedFunction] def elaborateFunction : TermElab := fun stx expected => do
  let parameter : Ident := ⟨stx[7]⟩
  let source ← sourceText stx.getArgs.back!
  let mut imports := []
  for binding in stx[2].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding
      | throwError "expected a named source import"
    let value ← elabTermEnsuringType value (mkConst ``String)
    imports := imports ++ [(name.getId.eraseMacroScopes.toString,← stringValue value)]
  let _ ← declarations (mkIdent `__veritile_imported_helper) parameter source imports
  let importValue ← mkListLit (mkApp2 (mkConst ``Prod [levelZero,levelZero]) (mkConst ``String) (mkConst ``String))
    (← imports.mapM fun (name,target) => mkAppM ``Prod.mk #[mkStrLit name,mkStrLit target])
  ensureHasType expected (mkApp3 (mkConst ``ImportedFiniteJitFunction.mk)
    (mkStrLit parameter.getId.eraseMacroScopes.toString) (mkStrLit source) importValue)

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifiers

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x => (identifiers x).any names.contains) then
        throwError "signed pointer arguments must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (identifiers counter).any names.contains then
      throwError "a loop cannot shadow a signed pointer argument"
  for arg in raw.getArgs do rejectWrites names arg

private partial def substitute (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident + $offset:tritonExpr) := raw then
    if let some (_,base) := values.find? (fun p => p.1==id.getId.eraseMacroScopes) then
      let offset : TSyntax `tritonExpr := ⟨← substitute values offset.raw⟩
      return (← `(tritonExpr| $id:ident + ($(($base:ident : Int)) + $offset:tritonExpr))).raw
  if let `(tritonExpr| $left:tritonExpr + $_:tritonExpr) := raw then
    if (identifiers left.raw).any (fun id => values.any (fun p => p.1==id)) then
      Macro.throwError "signed pointer binding requires one complete displacement"
  if let `(tritonExpr| $id:ident) := raw then
    if values.any (fun p => p.1==id.getId.eraseMacroScopes) then
      Macro.throwError "signed pointer binding requires a direct pointer plus a complete displacement"
  return raw.setArgs (← raw.getArgs.mapM (substitute values))

@[term_elab withFunctions] def elaborateInvocation : TermElab := fun stx expected => do
  let source ← sourceText stx.getArgs.back!
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete caller body"
  let mut metadata := #[]
  for binding in stx[12].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding
      | throwError "expected a named imported helper"
    let value ← elabTermEnsuringType value (mkConst ``ImportedFiniteJitFunction)
    let parameter ← stringValue (← mkAppM ``ImportedFiniteJitFunction.parameter #[value])
    let body ← stringValue (← mkAppM ``ImportedFiniteJitFunction.source #[value])
    let imports ← (← listValues (← mkAppM ``ImportedFiniteJitFunction.imports #[value])).mapM fun pair => do
      return (← stringValue (← mkAppM ``Prod.fst #[pair]),← stringValue (← mkAppM ``Prod.snd #[pair]))
    metadata := metadata ++ (← declarations name (mkIdent parameter.toName) body imports)
  unless !metadata.isEmpty do throwError "imported helper bindings cannot be empty"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut names := []
  let mut pointers := []
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      let name ← stringValue (← mkAppM ``Prod.fst #[pair])
      names:=names++[name]
      if field != ``CallSignature.scalars then pointers:=pointers++[name.toName]
  for constant in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    names:=names++[← stringValue constant]
  let _ ← liftMacroM <| FiniteHelperSource.prepare (metadata++stmts).toList names
  let stmts := (metadata++stmts).toList
  let mut offsets := []
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding
      | throwError "expected a named signed pointer offset"
    let name := id.getId.eraseMacroScopes
    unless pointers.contains name && !offsets.any (fun p => p.1==name) do
      throwError "signed offsets must name distinct pointer parameters"
    offsets:=offsets++[(name,← elabTermEnsuringType value (mkConst ``Int))]
  rejectWrites (offsets.map Prod.fst) parsed
  let used := identifiers parsed ++ identifiers stx
  let rec freeze (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::rest =>
      let mut fresh := Name.anonymous
      repeat
        let next ← mkFreshId
        fresh:=Name.mkSimple ("veritile_signed_pointer_"++(next.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains fresh do break
      withLetDecl fresh (mkConst ``Int) value fun bound => do
        let result ← freeze rest (values++[(name,mkIdent fresh)])
        mkLetFVars #[bound] result
    | [] =>
      let lowered : List (TSyntax `tritonStmt) ← liftMacroM <| stmts.mapM fun st => return ⟨← substitute values st.raw⟩
      let text := "{\n"++String.intercalate "\n" (lowered.map (render ·.raw))++"\n}"
      let raw := Syntax.atom .none text
      let call := mkNode ``CallSignature.withCall
        #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[14],stx[15],raw]
      PythonInlineSuites.elaborateInvocation call expected
  freeze offsets []

/-- Call typing prepends compile-time declarations. Imported helpers must still
precede those declarations for their existing lexical checker. Move only that
metadata prefix, never an executable statement or branch. -/
private def callMetadata (st : TSyntax `tritonStmt) : Bool :=
  match st with
  | `(tritonStmt| tl.call_signature)
  | `(tritonStmt| tl.ports inputs=[$_:term,*], outputs=[$_:term,*])
  | `(tritonStmt| tl.region $_:ident = $_:tritonDType $[, $_:ident = $_:tritonDType]*)
  | `(tritonStmt| tl.param $_:ident : $_:tritonDType) => Bool.true
  | _ => Bool.false

private def helperMetadata (st : TSyntax `tritonStmt) : Bool :=
  st.raw.getKind==``tritonSourceFiniteAlias || st.raw.getKind==``tritonSourceFiniteHelper

@[term_elab tritonBlock] def reorderCallMetadata : TermElab := fun stx expected => do
    let `(triton { $stmts:tritonStmt* }) := stx | throwUnsupportedSyntax
    let (header,rest) := stmts.toList.span callMetadata
    let (helpers,body) := rest.span helperMetadata
    if header.isEmpty || helpers.isEmpty then throwUnsupportedSyntax
    let reordered := (helpers++header++body).toArray
    let source ← liftMacroM `(triton { $reordered* })
    elabTerm source expected

end DSL.ImportedFiniteJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.ImportedFiniteJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceSyntax.importedParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``ImportedFiniteJitFunction do throwUnsupportedSyntax
  let imports := stx[2].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let parameters := stx[7].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let #[x] := parameters | throwError "imported unary helper requires one parameter"
  let body ← `(with_triton_imports [$imports,*] in triton_function ($x:ident) {})
  elaborateFunction (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.ImportedFiniteJitFunctions
