import VeriTile.Triton.DSL.NestedSpecializedJitFunctions

/-! Specialize a declared Nat constexpr whose value is a finite conditional
choice. Each branch is elaborated with its own exact Nat binding. The result
retains the original Lean condition and its decider; runtime Triton tests and
all floating primitives are delegated unchanged to the statement frontend. -/
namespace VeriTile.Triton.DSL.FiniteNatJitFunctions
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "static constexpr names required"
  return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)

private partial def letValue (value : Expr) : TermElabM Expr := do
  match value with
  | .fvar id=>
    match (← id.getDecl).value? with
    | some value=>letValue value
    | none=>return value
  | .mdata _ value=>letValue value
  | .letE _ _ value body _=>letValue (body.instantiate1 value)
  | _=>return value

/-- Inference follows TritonSemantic.to_tensor_type in
https://github.com/triton-lang/triton/blob/main/python/triton/language/semantic.py.
A bare Python integer binding has an integer type, including when a
tuple is expanded through temporaries. Floating tokens such as 0.0 keep their
original spelling and type. -/
private def integerBinding (value : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  let value := if value.raw.getKind==choiceKind then ⟨value.raw[0]!⟩ else value
  match value with
  | `(tritonExpr| $n:num) =>
    if n.getNat<2^31 then `(tritonExpr| tl.cast($n:num, tl.int32))
    else if n.getNat<2^32 then `(tritonExpr| tl.cast($n:num, tl.uint32))
    else if n.getNat<2^63 then `(tritonExpr| tl.cast($n:num, tl.int64))
    else Macro.throwErrorAt value "unsigned-64 Python integer bindings are not yet supported"
  | `(tritonExpr| -$n:num) =>
    if n.getNat≤2^31 then `(tritonExpr| tl.cast(-$n:num, tl.int32))
    else if n.getNat≤2^63 then `(tritonExpr| tl.cast(-$n:num, tl.int64))
    else Macro.throwErrorAt value "integer binding exceeds the supported signed-64 range"
  | _=>return value

private partial def lowerBindings (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lowerBindings raw[0]!
  let raw := raw.setArgs (← raw.getArgs.mapM lowerBindings)
  if raw.getKind==``GuardedJitFunctions.tupleValues && raw.getNumArgs==7 then
    let a ← integerBinding ⟨raw[4]!⟩
    let b ← integerBinding ⟨raw[6]!⟩
    return raw.setArgs (raw.getArgs.set! 4 a.raw |>.set! 6 b.raw)
  match raw with
  | `(tritonStmt| $name:ident = $value:tritonExpr) =>
    let value ← integerBinding value
    return (← `(tritonStmt| $name:ident = $value:tritonExpr)).raw
  | `(tritonStmt| $left:ident, $right:ident = $a:tritonExpr, $b:tritonExpr) =>
    let a ← integerBinding a
    let b ← integerBinding b
    return (← `(tritonStmt| $left:ident, $right:ident = $a:tritonExpr, $b:tritonExpr)).raw
  | _=>return raw

private def prepareFunctions (stx : Syntax) : TermElabM Syntax := do
  let mut bindings : Array (TSyntax `tritonCallBinding):=#[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $name:ident := $term:term) := binding | throwUnsupportedSyntax
    let value ← elabTerm term none
    unless (← inferType value).isConstOf ``SpecializedStatementJitFunction do throwUnsupportedSyntax
    let source ← whnf (← mkAppM ``SpecializedStatementJitFunction.source #[value])
    let .lit (.strVal source):=source | throwError "static helper source required"
    let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize
      (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
    let raw ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
      | .ok raw=>pure raw | .error message=>throwError "{message}"
    let raw ← liftMacroM (lowerBindings raw)
    let `(triton { $body:tritonStmt* }) := raw | throwError "complete helper body required"
    let source:=String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))
    let rebuilt:=mkApp3 (mkConst ``SpecializedStatementJitFunction.mk)
      (← mkAppM ``SpecializedStatementJitFunction.parameters #[value])
      (← mkAppM ``SpecializedStatementJitFunction.constexpr #[value]) (mkStrLit source)
    let rebuilt ← exprToSyntax rebuilt
    bindings:=bindings.push (← `(tritonCallBinding| $name:ident := $rebuilt:term))
  let profile : TSyntax `term:=⟨stx[1]⟩
  let arguments:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let rebuilt ← `(with_triton_call $profile:term call_args [$arguments,*]
    jit_functions [$bindings,*] in triton {})
  return stx.setArgs (stx.getArgs.set! 8 rebuilt.raw[8]!)

private partial def elaborateChoices (stx : Syntax) (expected : Option Expr)
    (remaining : List Name) : TermElabM Expr := do
  let name::tail:=remaining
    | return ← NestedSpecializedJitFunctions.elaborateInvocation stx expected
  let bound ← elabTerm (mkIdent name) none
  unless ← isDefEq (← inferType bound) (mkConst ``Nat) do
    return ← elaborateChoices stx expected tail
  let value ← letValue bound
  unless value.isAppOfArity ``ite 5 do
    return ← elaborateChoices stx expected tail
  let arguments:=value.getAppArgs
  let branch (choice : Expr) : TermElabM Expr :=
    withLetDecl name (mkConst ``Nat) choice fun binding=>do
      let result ← elaborateChoices stx (some (mkConst ``ComputeKernel)) tail
      mkLetFVars #[binding] result
  let yes ← branch arguments[3]!
  let no ← branch arguments[4]!
  -- Nat and ComputeKernel are both in Type, so the original ite universe
  -- instantiation is retained together with the actual condition/decider.
  ensureHasType expected (mkAppN value.getAppFn
    #[mkConst ``ComputeKernel,arguments[1]!,arguments[2]!,yes,no])

@[term_elab StatementJitFunctions.withFunctions]
def elaborateInvocation : TermElab := fun stx expected=>do
  let stx ← prepareFunctions stx
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut names:=[]
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "static constexpr names required"
    names:=names++[name.toName]
  elaborateChoices stx expected names
end VeriTile.Triton.DSL.FiniteNatJitFunctions
