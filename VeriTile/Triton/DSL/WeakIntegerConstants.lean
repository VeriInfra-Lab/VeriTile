import VeriTile.Triton.DSL.IntegralBooleanOperands

/-! A constexpr integer next to an explicitly FP32 tensor is materialized at
the operation's FP32 type. It is not a runtime integer conversion. This is the
scalar-constant branch of Triton 3.7.1 `binary_op_type_checking_impl` (lines
161–191 of python/triton/language/semantic.py). Runtime integer parameters
and tensor registers keep their existing conversion operations. -/
namespace VeriTile.Triton.DSL.WeakIntegerConstants
open Lean Elab Term Meta

private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)

private partial def explicitFP32 (value : TSyntax `tritonExpr) : Bool :=
  if let some (_,dtype):=methodCast? value then
    match dtype with | `(tritonDType| tl.float32)=>Bool.true | _=>Bool.false
  else match value with
  | `(tritonExpr| ($value:tritonExpr))=>explicitFP32 value
  | `(tritonExpr| tl.cast($_:tritonExpr, tl.float32))=>Bool.true
  | _=>Bool.false

private partial def constant? (constants : List (Name × TSyntax `tritonExpr))
    (value : TSyntax `tritonExpr) : Option (TSyntax `tritonExpr) :=
  match value with
  | `(tritonExpr| $name:ident)=>constants.lookup name.getId.eraseMacroScopes
  | `(tritonExpr| ($value:tritonExpr))=>constant? constants value
  | _=>none

private partial def lower (constants : List (Name × TSyntax `tritonExpr))
    (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (lower constants))
  let operands:=match raw with
    | `(tritonExpr| $left:tritonExpr / $right:tritonExpr)=>some (left,right,0)
    | `(tritonExpr| $left:tritonExpr * $right:tritonExpr)=>some (left,right,1)
    | `(tritonExpr| $left:tritonExpr + $right:tritonExpr)=>some (left,right,2)
    | `(tritonExpr| $left:tritonExpr - $right:tritonExpr)=>some (left,right,3)
    | _=>none
  let some (left,right,operator):=operands | return raw
  let left:=if explicitFP32 right then (constant? constants left).getD left else left
  let right:=if explicitFP32 left then (constant? constants right).getD right else right
  let result ← match operator with
    | 0=>`(tritonExpr| $left:tritonExpr / $right:tritonExpr)
    | 1=>`(tritonExpr| $left:tritonExpr * $right:tritonExpr)
    | 2=>`(tritonExpr| $left:tritonExpr + $right:tritonExpr)
    | _=>`(tritonExpr| $left:tritonExpr - $right:tritonExpr)
  return result.raw

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]!::(← listValues value.getAppArgs[2]!)
  throwError "weak constants require static constexpr metadata"

def prepare (stx : Syntax) : TermElabM Syntax := do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut constants:List (Name × TSyntax `tritonExpr):=[]
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "static constexpr names required"
    let id:=mkIdent name.toName
    let term ← elabTerm id none
    if ← isDefEq (← inferType term) (mkConst ``Nat) then
      let constant ← liftMacroM `(tritonExpr| $(($id:ident : ℝ)))
      constants:=constants++[(name.toName,constant)]
  if constants.isEmpty then return stx
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure (first value) | .error message=>throwError "{message}"
  let changed ← liftMacroM (lower constants parsed)
  let `(triton { $body:tritonStmt* }):=changed | throwError "complete scalar-promotion source required"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

end VeriTile.Triton.DSL.WeakIntegerConstants
