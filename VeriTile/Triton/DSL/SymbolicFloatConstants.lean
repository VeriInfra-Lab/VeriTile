import VeriTile.Triton.DSL.PythonTensorTuples

/-! Symbolic FP32 materialization of immutable Python constexpr scalars.
Their numerical values are supplied by the invocation's scalar materialization
interface. This introduces scalar bindings, never a tensor/result callback.
The original signature still records that these arguments are constexprs.
Host-to-FP32 literal realization is a separate, explicit IO obligation. -/
namespace VeriTile.Triton.DSL.SymbolicFloatConstants
open Lean Elab Term Meta

def profile (signature : CallSignature) (names : List String) : CallSignature :=
  {signature with
    constexpr:=signature.constexpr.filter (fun name=>!names.contains name)
    scalars:=signature.scalars++names.map (fun name=>(name,.fp32))}

syntax (name := symbolicCall) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " fp32_constants " "[" ident,* "]"
  " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]!::(← listValues value.getAppArgs[2]!)
  throwError "symbolic constant names require a static signature"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  if let `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr):=raw then
    if names.contains name.getId.eraseMacroScopes then
      throwError "symbolic constexpr arguments must remain immutable"
  let args:=raw.getArgs.toList
  let leaf:=raw.getKind.getString!
  -- Intrinsic keyword arguments do not assign to local variables.
  if leaf.startsWith "tritonStmt" || leaf.startsWith "tuple" then
    for operator in ["=",":=","+=","-=","*=","/=","//=","%=","&=","|=","^="] do
      if let some i:=args.findIdx? (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString==operator) then
        if (args.take i).any (fun arg=>(ids arg).any names.contains) then
          throwError "symbolic constexpr arguments must remain immutable"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then
      throwError "a loop cannot shadow a symbolic constexpr argument"
  for child in raw.getArgs do rejectWrites names child

@[term_elab symbolicCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let signature ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let names:=stx[8].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let mut declared:List Name:=[]
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[signature]) do
    let .lit (.strVal name) ← whnf value | throwError "static constexpr names required"
    declared:=declared++[name.toName]
  unless !names.isEmpty && names.eraseDups==names && names.all declared.contains do
    throwError "symbolic FP32 constants must be distinct declared constexpr parameters"
  let signatureExpr ← mkAppM ``profile #[signature,
    ← mkListLit (mkConst ``String) (names.map (mkStrLit ∘ Name.toString))]
  let signatureSyntax ← exprToSyntax signatureExpr
  let arguments:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let base ← `(with_triton_call $signatureSyntax:term call_args [$arguments,*] in triton {})
  let base:=base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) stx.getArgs.back!)
  let base ← PythonTensorTuples.prepare base
  let source ← match RawSourceText.normalize base.getArgs.back!.getAtomVal with
    | .ok value=>pure value | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwError "{message}"
  rejectWrites names parsed
  PythonTensorTuples.elaborateInvocation base expected

end VeriTile.Triton.DSL.SymbolicFloatConstants
