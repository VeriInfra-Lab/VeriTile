import VeriTile.Triton.DSL.IntegerCallConditionals

/-! FP32 value binding at the invocation boundary. Elaborate the unchanged body
against its runtime scalar interface, then instantiate only immutable scalar
references with precision-scoped literals. This retains FP32 argument conversion
in Boolean comparisons as well as floating arithmetic. No statement is inserted,
deleted, reordered or selected from a second kernel implementation. -/
namespace VeriTile.Triton.DSL.FP32CallValues
open Lean Elab Term Meta
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "call signature must be a static list"
private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifiers
private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  if raw.getKind==choiceKind then return ← rejectWrites names raw[0]!
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/=","//=","%=","&=","|=","^=","<<=",">>="] do
    if let some index := args.findIdx? (fun token => token.isAtom && token.getAtomVal.trimAscii.toString==operator) then
      if ((args.take index).flatMap identifiers).any names.contains then
        throwError "FP32 call values must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for","tl.static_range"].contains first.getAtomVal.trimAscii.toString &&
        (identifiers counter).any names.contains then throwError "loop cannot shadow an FP32 call value"
  for child in raw.getArgs do rejectWrites names child

@[term_elab IntegerCallArguments.withIntegers]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut fp32Names : List String := []
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    if dtype.isConstOf ``ScalarType.fp32 then
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair])
        | throwError "scalar name must be static"
      fp32Names := fp32Names ++ [name]
  let mut values : List (String × Expr) := []
  let mut bindings : Array (TSyntax `tritonCallBinding) := #[]
  for item in stx[4].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := item
      | throwError "named call binding required"
    if fp32Names.contains name.getId.eraseMacroScopes.toString then
      if values.any (·.1==name.getId.eraseMacroScopes.toString) then throwError "duplicate FP32 call binding"
      let value ← elabTermEnsuringType value (mkConst ``Real)
      values := values ++ [(name.getId.eraseMacroScopes.toString,value)]
    else bindings := bindings.push ⟨item⟩
  let signature : TSyntax `term := ⟨stx[1]⟩
  let integers := stx[8].getSepArgs.map (fun x => (⟨x⟩ : TSyntax `tritonCallBinding))
  let sizes := stx[12].getSepArgs.map (fun x => (⟨x⟩ : TSyntax `tritonCallBinding))
  let adjusted ← `(with_triton_call $signature call_args [$bindings,*]
    integer_args [$integers,*] arange_args [$sizes,*] in triton {})
  let adjusted := adjusted.raw.setArgs (adjusted.raw.getArgs.set! (adjusted.raw.getArgs.size-1) stx.getArgs.back!)
  let adjusted ← PythonConditionGrouping.prepareSource adjusted
  let source := adjusted.getArgs.back!.getAtomVal
  let body ← match Parser.runParserCategory (← getEnv) `term ("triton "++source) with
    | .ok body => pure body | .error message => throwError "{message}"
  rejectWrites (values.map (·.1.toName)) body
  let result ← IntegerCallSyntax.elaborateInvocation adjusted expected
  let mut replacements : List (String × Expr) := []
  for (name,value) in values do
    let literal ← mkAppM ``Op.const #[value]
    let typedValue ← mkAppM ``Op.withPrecision #[mkConst ``FloatDType.real,
      mkConst ``IndexComputePrecision.fp32,literal]
    replacements := replacements ++ [(name,typedValue)]
  let mut result := result
  for (name,value) in replacements do
    result := result.replace fun e => do
      unless e.isAppOfArity ``Op.ref 3 do none
      let args := e.getAppArgs
      unless args[0]!.isConstOf ``TileDType.real && args[1]!.isAppOf ``List.nil do none
      let .lit (.strVal found) := args[2]! | none
      if found==name then some value else none
  return result
end VeriTile.Triton.DSL.FP32CallValues
