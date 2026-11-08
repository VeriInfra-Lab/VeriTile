import VeriTile.Triton.DSL.ModuleFloatConstants
import VeriTile.Triton.DSL.StatementJitFunctions
import VeriTile.Triton.DSL.OptionalStringConstants
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-! Closed source `math.sqrt`/`math.pi` module expressions, bound outside a
pasted body. These are source tags: Real interpretations use the corresponding
mathematical expression; a native interpretation must separately bind the
actual rounded host constant. No exact host-to-Real equality is asserted. -/
namespace VeriTile.Triton.DSL.ModuleMathConstants
open Lean Elab Term Meta

private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "module math bindings must be static"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "module math bindings must be static"
private partial def expression (raw : TSyntax `tritonExpr) : TermElabM (TSyntax `term) := do
  if raw.raw.getKind==choiceKind then return ← expression ⟨raw.raw[0]!⟩
  if raw.raw.getKind==``StatementJitFunctions.helperCall && raw.raw[0].isIdent &&
      raw.raw[0].getId.eraseMacroScopes==`math.sqrt then
    let args:=raw.raw[2].getSepArgs
    unless args.size==1 do throwError "module math sqrt requires one argument"
    let argument : TSyntax `StatementJitFunctions.statementJitArgument:=⟨args[0]!⟩
    let `(statementJitArgument| $inner:tritonExpr):=argument
      | throwError "module math sqrt requires one positional argument"
    let inner ← expression inner
    return ← `(Real.sqrt $inner)
  match raw with
  | `(tritonExpr| $n:num)=>`(($n:num : ℝ))
  | `(tritonExpr| $n:scientific)=>`(($n:scientific : ℝ))
  | `(tritonExpr| math.pi)=>`(Real.pi)
  | `(tritonExpr| math.sqrt($value:tritonExpr))=>do
    let value ← expression value
    `(Real.sqrt $value)
  | `(tritonExpr| ($value:tritonExpr))=>expression value
  | `(tritonExpr| -$value:tritonExpr)=>do
    let value ← expression value
    `(-$value)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)=>do
    let a ← expression a;let b ← expression b
    `($a + $b)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)=>do
    let a ← expression a;let b ← expression b
    `($a - $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)=>do
    let a ← expression a;let b ← expression b
    `($a * $b)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr)=>do
    let a ← expression a;let b ← expression b
    `($a / $b)
  | _=>throwError "module math binding requires a closed numeric/sqrt/pi expression"
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i:=args.findIdx? (fun x=>x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x=>(ids x).any names.contains) then
        throwError "module math constants cannot be reassigned"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then throwError "a loop cannot shadow a module math constant"
  for child in raw.getArgs do rejectWrites names child
def bindMath (profile : CallSignature) (names : List String) : CallSignature :=
  { profile with
    sourceConstants := profile.sourceConstants.filter (fun p=> !names.contains p.1)
    constexpr := profile.constexpr ++ names }

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut values : List (Name × TSyntax `term):=[]
  let mut moduleNames : List String:=[]
  for pair in ← listValues (← mkAppM ``CallSignature.sourceConstants #[profile]) do
    let name ← textValue (← mkAppM ``Prod.fst #[pair])
    let source ← textValue (← mkAppM ``Prod.snd #[pair])
    moduleNames:=moduleNames++[name]
    if source.startsWith "math." then
      discard <| liftMacroM <| SourceBindings.constant (← getEnv) name "0"
      let parsed ← match Parser.runParserCategory (← getEnv) `tritonExpr source with
        | .ok parsed=>pure parsed | .error message=>throwError "{message}"
      let value ← expression ⟨parsed⟩
      values:=values++[(name.toName,value)]
  if values.isEmpty then return ← OptionalStringConstants.elaborateInvocation (← ModuleFloatConstants.prepareSource stx) expected
  let mut parameters : List String:=[]
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      parameters:=parameters++[← textValue (← mkAppM ``Prod.fst #[pair])]
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    parameters:=parameters++[← textValue value]
  for pair in ← listValues (← mkAppM ``CallSignature.unaryImports #[profile]) do
    moduleNames:=moduleNames++[← textValue (← mkAppM ``Prod.fst #[pair])]
  unless moduleNames.eraseDups==moduleNames && moduleNames.all (fun name=> !parameters.contains name) do
    throwError "module constants must be distinct from parameters and each other"
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwError "{message}"
  rejectWrites (values.map Prod.fst) parsed
  let names : Array (TSyntax `term):=(values.map (fun p=>⟨Syntax.mkStrLit p.1.toString⟩)).toArray
  let signature : TSyntax `term:=⟨stx[1]⟩
  let signature ← `(bindMath $signature [$names,*])
  let adjusted:=stx.setArgs (stx.getArgs.set! 1 signature)
  let rec freeze (todo : List (Name × TSyntax `term)) : TermElabM Expr := do
    match todo with
    | []=>OptionalStringConstants.elaborateInvocation (← ModuleFloatConstants.prepareSource adjusted) expected
    | (name,value)::rest=>
      let value ← elabTermEnsuringType value (mkConst ``Real)
      withLetDecl name (mkConst ``Real) value fun bound=>do
        let result ← freeze rest
        mkLetFVars #[bound] result
  freeze values
end VeriTile.Triton.DSL.ModuleMathConstants
