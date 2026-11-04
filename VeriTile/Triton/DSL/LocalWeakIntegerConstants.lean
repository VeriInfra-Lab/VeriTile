import VeriTile.Triton.DSL.StatementJitFunctions

/-! Immutable local integer constexprs adjacent to explicit FP32 tensors use
FP32 literal materialization. This opt-in preparation leaves integer uses,
parameter metadata and the original pasted function object unchanged. -/
namespace VeriTile.Triton.DSL.LocalWeakIntegerConstants
open Lean Elab Term Meta

private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def fp32 (value : TSyntax `tritonExpr) : Bool :=
  if let some (_,dtype):=methodCast? value then
    match dtype with | `(tritonDType| tl.float32)=>Bool.true | _=>Bool.false
  else match value with
    | `(tritonExpr| ($value:tritonExpr))=>fp32 value
    | `(tritonExpr| tl.cast($_:tritonExpr, tl.float32))=>Bool.true
    | _=>Bool.false
private partial def natural? (value : TSyntax `tritonExpr) : Option Nat :=
  match value with
  | `(tritonExpr| $number:num)=>some number.getNat
  | `(tritonExpr| ($value:tritonExpr))=>natural? value
  | _=>none
private partial def lower (constants : List (Name × TSyntax `tritonExpr))
    (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (lower constants))
  let operands:=match raw with
    | `(tritonExpr| $x:tritonExpr / $y:tritonExpr)=>some (x,y,0)
    | `(tritonExpr| $x:tritonExpr * $y:tritonExpr)=>some (x,y,1)
    | `(tritonExpr| $x:tritonExpr + $y:tritonExpr)=>some (x,y,2)
    | `(tritonExpr| $x:tritonExpr - $y:tritonExpr)=>some (x,y,3)
    | _=>none
  let some (x,y,op):=operands | return raw
  let substitute (value : TSyntax `tritonExpr) :=
    match value with
    | `(tritonExpr| $name:ident)=>(constants.lookup name.getId.eraseMacroScopes).getD value
    | _=>value
  let x:=if fp32 y then substitute x else x
  let y:=if fp32 x then substitute y else y
  let result ← match op with
    | 0=>`(tritonExpr| $x:tritonExpr / $y:tritonExpr)
    | 1=>`(tritonExpr| $x:tritonExpr * $y:tritonExpr)
    | 2=>`(tritonExpr| $x:tritonExpr + $y:tritonExpr)
    | _=>`(tritonExpr| $x:tritonExpr - $y:tritonExpr)
  return result.raw

private def source (text : String) : TermElabM String := do
  let base ← `(with_triton_call ({} : CallSignature) call_args [] in triton {})
  let prepared ← PythonFullConstructors.prepareSource
    (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) (Syntax.atom .none ("{\n"++text++"\n}"))))
  let normalized ← match RawSourceText.normalize prepared.getArgs.back!.getAtomVal with
    | .ok value=>pure value | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++normalized++"\n}") with
    | .ok value=>pure (first value) | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete helper body required"
  let mut constants:List (Name × TSyntax `tritonExpr):=[]
  let mut result:Array (TSyntax `tritonStmt):=#[]
  for statement in body do
    if let `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr):=statement then
      let key:=name.getId.eraseMacroScopes
      if constants.any (·.1==key) then throwError "local constexpr cannot be redefined"
      if let some number:=natural? value then
        let literal ← match Parser.runParserCategory (← getEnv) `tritonExpr (toString number++".0") with
          | .ok value=>pure (first value) | .error message=>throwError "{message}"
        constants:=constants++[(key,⟨literal⟩)]
      result:=result.push statement
      continue
    let args:=statement.raw.getArgs.toList
    for operator in ["=",":=","+=","-=","*=","/="] do
      if let some i:=args.findIdx? (fun arg=>arg.isAtom && arg.getAtomVal.trimAscii.toString==operator) then
        if (args.take i).any (fun arg=>(ids arg).any (fun name=>constants.any (·.1==name))) then
          throwError "local constexpr must remain immutable"
    result:=result.push ⟨← liftMacroM (lower constants statement.raw)⟩
  return String.intercalate "\n" (result.toList.map (PythonFullConstructors.render ·.raw))

/-- Prepare only temporary helper values used during this invocation. -/
def prepare (stx : Syntax) : TermElabM Syntax := do
  let mut bindings:Array (TSyntax `tritonCallBinding):=#[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term):=binding | throwError "named statement helpers required"
    let f ← elabTermEnsuringType value (mkConst ``StatementJitFunction)
    let .lit (.strVal original) ← whnf (← mkAppM ``StatementJitFunction.source #[f]) |
      throwError "static helper source required"
    let changed ← source original
    let f ← mkAppM ``StatementJitFunction.mk #[
      ← mkAppM ``StatementJitFunction.parameters #[f],
      ← mkAppM ``StatementJitFunction.constexpr #[f],mkStrLit changed]
    let value ← exprToSyntax f
    bindings:=bindings.push (← `(tritonCallBinding| $name:ident := $value:term))
  let signatureSyntax:TSyntax `term:=⟨stx[1]⟩
  let arguments:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let base ← `(with_triton_call $signatureSyntax:term call_args [$arguments,*] jit_functions [$bindings,*] in triton {})
  return base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) stx.getArgs.back!)
end VeriTile.Triton.DSL.LocalWeakIntegerConstants
