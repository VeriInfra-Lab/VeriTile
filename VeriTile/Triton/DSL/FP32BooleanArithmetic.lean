import VeriTile.Triton.DSL.PythonInlineSuites

/-! Numeric Boolean multiplication for straight-line FP32 source. A Boolean
multiplicand becomes the exact FP32 value 0 or 1; the floating multiplication
remains. Selecting between the value and zero would lose NaNs and signed-zero
behavior. Unsupported operand types and control-flow merges fail closed. -/
namespace VeriTile.Triton.DSL.FP32BooleanArithmetic
open Lean Elab Term Meta

private inductive Kind where
  | fp32 | boolean | integer | pointer | unknown
  deriving BEq, Inhabited
private abbrev Types := List (Name × Kind)

private def arithmetic (a b : Kind) : Kind :=
  if a==.fp32 && (b==.fp32 || b==.integer || b==.boolean) ||
      b==.fp32 && (a==.integer || a==.boolean) then .fp32
  else if a==.integer && b==.integer then .integer else .unknown

private partial def kind (types : Types) (source : TSyntax `tritonExpr) : Kind :=
  match source with
  | `(tritonExpr| $id:ident) => (types.lookup id.getId.eraseMacroScopes).getD .unknown
  | `(tritonExpr| $n:num) => .integer
  | `(tritonExpr| $n:scientific) => .fp32
  | `(tritonExpr| true) | `(tritonExpr| false) => .boolean
  | `(tritonExpr| ($value:tritonExpr)) => kind types value
  | `(tritonExpr| -$value:tritonExpr) => kind types value
  | `(tritonExpr| not $value:tritonExpr) => if kind types value==.boolean then .boolean else .unknown
  | `(tritonExpr| $_:tritonExpr < $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr <= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr > $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr >= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr == $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr != $_:tritonExpr) => .boolean
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
    let ka:=kind types a; let kb:=kind types b
    if ka==.pointer && kb==.integer || ka==.integer && kb==.pointer then .pointer
    else arithmetic ka kb
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
    let ka:=kind types a; let kb:=kind types b
    if ka==.pointer && kb==.integer then .pointer else arithmetic ka kb
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => arithmetic (kind types a) (kind types b)
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr) =>
    let ka:=kind types a; let kb:=kind types b
    if ka==kb then ka else .unknown
  | `(tritonExpr| tl.program_id($_:tritonExpr))
  | `(tritonExpr| tl.program_id(axis=$_:tritonExpr))
  | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr)) => .integer
  | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) =>
    if kind types p==.pointer then .fp32 else .unknown
  | `(tritonExpr| tl.where($_:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
    let ka:=kind types a; let kb:=kind types b
    if ka==kb then ka else arithmetic ka kb
  | _ => .unknown

private partial def lower (types : Types) (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM (lower types))
  if let `(tritonExpr| $a:tritonExpr * $b:tritonExpr) := raw then
    let ka:=kind types a; let kb:=kind types b
    if ka==.unknown || kb==.unknown then
      Macro.throwErrorAt raw "multiplication operand needs a supported explicit FP32/integer source type"
    if ka==.boolean && kb==.fp32 then
      return (← `(tritonExpr| tl.where($a, 1.0, 0.0) * $b)).raw
    if ka==.fp32 && kb==.boolean then
      return (← `(tritonExpr| $a * tl.where($b, 1.0, 0.0))).raw
    if ka==.boolean || kb==.boolean then
      Macro.throwErrorAt raw "numeric Boolean multiplication currently requires an FP32 operand"
  return raw

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "FP32 Boolean call profile must have static parameter names"

private def initialTypes (profileSyntax : Syntax) : TermElabM Types := do
  let profile ← elabTermEnsuringType profileSyntax (mkConst ``Triton.CallSignature)
  let mut types : Types := []
  for field in [``Triton.CallSignature.pointers,``Triton.CallSignature.floatingPointers,``Triton.CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair])
        | throwError "call parameter name must be literal"
      let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
      let .const dtypeName _ := dtype.getAppFn
        | throwError "numeric Boolean frontend requires a statically known FP32 profile"
      let classification ←
        if field==``Triton.CallSignature.scalars then
          if dtypeName==``Triton.ScalarType.fp32 then pure Kind.fp32
          else if [``Triton.ScalarType.integer,``Triton.ScalarType.i32,``Triton.ScalarType.i64].contains dtypeName then pure Kind.integer
          else throwError "numeric Boolean frontend requires FP32 floating scalar parameters"
        else
          if [``Triton.SourceDType.fp32,``Triton.FloatDType.fp32].contains dtypeName then pure Kind.pointer
          else throwError "numeric Boolean frontend requires FP32 pointer storage"
      types:=types++[(Name.mkSimple name,classification)]
  for value in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent (Name.mkSimple name)) none
    let dtype ← inferType value
    let classification ←
      if ← isDefEq dtype (mkConst ``Nat) then pure Kind.integer
      else if ← isDefEq dtype (mkConst ``Int) then pure Kind.integer
      else if ← isDefEq dtype (mkConst ``Bool) then pure Kind.boolean
      else throwError "unsupported constexpr type in numeric Boolean frontend"
    types:=types++[(Name.mkSimple name,classification)]
  return types

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ k args => if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""; let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token; previous:=token
  return out

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected complete FP32 Boolean body"
  let mut types ← initialTypes stx[1]
  let mut body : List Syntax:=[]
  for statement in statements do
    match statement with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      let lowered ← liftMacroM <| lower types value.raw
      let lowered : TSyntax `tritonExpr := ⟨lowered⟩
      body:=body++[(← `(tritonStmt| $id:ident = $lowered:tritonExpr)).raw]
      let name:=id.getId.eraseMacroScopes
      types:=(name,kind types value)::types.filter (fun p => p.1 != name)
    | `(tritonStmt| tl.store($ptr:tritonExpr, $value:tritonExpr))
    | `(tritonStmt| tl.store($ptr:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
      body:=body++[← liftMacroM <| lower types statement.raw]
    | _ => throwErrorAt statement "numeric Boolean frontend currently requires straight-line assignments and stores"
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (body.map render)++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  PythonInlineSuites.elaborateSignature (← prepare stx) expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  PythonInlineSuites.elaborateInvocation (← prepare stx) expected

end VeriTile.Triton.DSL.FP32BooleanArithmetic
