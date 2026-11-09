import VeriTile.Triton.DSL.ReboundEntryConstants

/-! Explicit integer dtype constexprs are invocation metadata. Resolve only
statically known integer SourceDType constructors in dtype positions; preserve
every original constructor and cast. Unsupported values and writes fail. -/
namespace VeriTile.Triton.DSL.IntegerDTypeCallConstants
open Lean Elab Term Meta
syntax (name := integerDTypeName) (priority := low) ident : tritonDType

def withoutTypes (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with constexpr := profile.constexpr.filter (!names.contains ·)}
private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "integer dtype names must be static"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "integer dtype names must be static"
private def dtypeSyntax (value : Expr) : TermElabM (TSyntax `tritonDType) := do
  let value ← whnf value
  if value.isConstOf ``SourceDType.i8 then return ← `(tritonDType| tl.int8)
  if value.isConstOf ``SourceDType.i16 then return ← `(tritonDType| tl.int16)
  if value.isConstOf ``SourceDType.i32 then return ← `(tritonDType| tl.int32)
  if value.isConstOf ``SourceDType.i64 then return ← `(tritonDType| tl.int64)
  if value.isConstOf ``SourceDType.u8 then return ← `(tritonDType| tl.uint8)
  if value.isConstOf ``SourceDType.u16 then return ← `(tritonDType| tl.uint16)
  if value.isConstOf ``SourceDType.u32 then return ← `(tritonDType| tl.uint32)
  if value.isConstOf ``SourceDType.u64 then return ← `(tritonDType| tl.uint64)
  throwError "integer dtype constexpr requires a known integer SourceDType constructor"
private partial def lower (values : List (Name × TSyntax `tritonDType)) (raw : Syntax) : TermElabM Syntax := do
  if raw.getKind==choiceKind then return ← lower values raw[0]!
  if let `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $key:ident=$value:term)):=raw then
    if key.getId.eraseMacroScopes==`dtype && value.raw.isIdent then
      if let some (_,dt):=values.find? (·.1==value.raw.getId.eraseMacroScopes) then
        return (← liftMacroM `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $dt:tritonDType))).raw
  if raw.getKind==``integerDTypeName then
    let name:=raw[0].getId.eraseMacroScopes
    let some (_,value):=values.find? (·.1==name) | throwError "named integer dtype requires a declared SourceDType constexpr"
    return value.raw
  if raw.isIdent && values.any (·.1==raw.getId.eraseMacroScopes) then
    throwError "integer dtype constexpr may occur only in dtype positions"
  return raw.setArgs (← raw.getArgs.mapM (lower values))

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut values : List (Name × TSyntax `tritonDType):=[]
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let name ← textValue entry
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``SourceDType) then
      values:=values++[(name.toName,← dtypeSyntax value)]
  if values.isEmpty then return ← ReboundEntryConstants.elaborateInvocation stx expected
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):= ← lower values parsed | throwError "complete source body required"
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let names : Array (TSyntax `term):=(values.map (fun p=>⟨Syntax.mkStrLit p.1.toString⟩)).toArray
  let signature : TSyntax `term:=⟨stx[1]⟩
  let signature ← `(withoutTypes $signature [$names,*])
  let adjusted:=stx.setArgs ((stx.getArgs.set! 1 signature).set! (stx.getArgs.size-1) raw)
  ReboundEntryConstants.elaborateInvocation adjusted expected
end VeriTile.Triton.DSL.IntegerDTypeCallConstants
