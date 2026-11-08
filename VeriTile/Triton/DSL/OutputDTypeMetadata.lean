import VeriTile.Triton.DSL.PythonShapeExpressions

/-! Python's OUT_DTYPE constexpr is caller type metadata. Its existing DSL
constructor/cast spelling is specialized for the actual caller dtype before
passing the complete source to the shape frontend.
No runtime instruction or cast is removed. -/
namespace VeriTile.Triton.DSL.OutputDTypeMetadata
open Lean Elab Term Meta

def withoutOutputType (profile : CallSignature) : CallSignature :=
  { profile with constexpr := profile.constexpr.filter (· != "OUT_DTYPE") }

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "output dtype metadata requires literal constexpr names"

private partial def lower (dtype : TSyntax `tritonDType) (raw : Syntax) : TermElabM Syntax := do
  if let `(tritonDType| OUT_DTYPE) := raw then return dtype.raw
  if (raw.isIdent && (raw.getId.eraseMacroScopes == `OUT_DTYPE ||
      raw.getId.eraseMacroScopes.toString.startsWith "OUT_DTYPE.")) ||
      (raw.isAtom && raw.getAtomVal == "OUT_DTYPE") then
    throwErrorAt raw "OUT_DTYPE may occur only in a constructor or cast dtype position"
  return raw.setArgs (← raw.getArgs.mapM (lower dtype))

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

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let constants ← (← listValues (← mkAppM ``CallSignature.constexpr #[profile])).mapM fun value => do
    let .lit (.strVal name) ← whnf value | throwError "constexpr names must be literal"
    return name
  unless constants.contains "OUT_DTYPE" do
    return ← PythonShapeExpressions.elaborateInvocation stx expected
  unless (constants.filter (· == "OUT_DTYPE")).length == 1 do
    throwError "OUT_DTYPE must be declared exactly once"
  for field in [``CallSignature.pointers, ``CallSignature.floatingPointers,
      ``CallSignature.scalars, ``CallSignature.sourceConstants, ``CallSignature.unaryImports] do
    for entry in ← listValues (← mkAppM field #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry])
        | throwError "call signature names must be literal"
      if name == "OUT_DTYPE" then throwError "OUT_DTYPE must be a distinct constexpr parameter"
  let argument ← elabTerm (mkIdent `OUT_DTYPE) (some (mkConst ``FloatDType))
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw := prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton " ++ raw.getAtomVal) with
    | .ok parsed => pure parsed
    | .error message => throwErrorAt raw "{message}"
  let adjusted ← exprToSyntax (← mkAppM ``withoutOutputType #[profile])
  let variants : List (TSyntax `tritonDType) := [← `(tritonDType| tl.float64),
    ← `(tritonDType| tl.float32), ← `(tritonDType| tl.float16),
    ← `(tritonDType| tl.bfloat16), ← `(tritonDType| tl.float8e4nv), ← `(tritonDType| tl.float8e5)]
  let elaborateVariant (dtype : TSyntax `tritonDType) : TermElabM Expr := do
    let lowered ← lower dtype parsed
    let `(triton { $statements:tritonStmt* }) := lowered | throwError "expected a complete typed source body"
    let raw := Syntax.atom .none ("{\n" ++ String.intercalate "\n" (statements.toList.map (fun s => render s.raw)) ++ "\n}")
    let args := (stx.getArgs.set! 1 adjusted.raw).set! (stx.getArgs.size - 1) raw
    PythonShapeExpressions.elaborateInvocation (stx.setArgs args) expected
  let constructors := [``FloatDType.real, ``FloatDType.fp32, ``FloatDType.fp16,
    ``FloatDType.bf16, ``FloatDType.f8e4, ``FloatDType.f8e5]
  let known ← whnf argument
  if let some index := constructors.findIdx? known.isConstOf then
    return ← elaborateVariant variants[index]!
  let branches ← variants.mapM elaborateVariant
  let motive := mkLambda `dtype BinderInfo.default (mkConst ``FloatDType) (mkConst ``ComputeKernel)
  mkAppOptM ``FloatDType.casesOn (#[some motive,some argument] ++ branches.toArray.map some)
end VeriTile.Triton.DSL.OutputDTypeMetadata
