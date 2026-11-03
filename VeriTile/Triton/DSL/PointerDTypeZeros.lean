import VeriTile.Triton.DSL.ScalarInt64StrideWhile

/-! Resolve a zeros constructor's explicit pointer-element dtype from its
invocation signature. The pasted body, constructor shape and initial value are
unchanged. Floating precision is retained by the ordinary typed constructor. -/
namespace VeriTile.Triton.DSL.PointerDTypeZeros
open Lean Elab Term Meta
private partial def pairs (e : Expr) : TermElabM (List (Name × TSyntax `tritonDType)) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  unless e.isAppOfArity ``List.cons 3 do throwError "static pointer signature required"
  let pair := e.getAppArgs[1]!
  let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair]) | throwError "literal pointer name required"
  let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
  let dtype ← if dtype.isConstOf ``SourceDType.fp32 then `(tritonDType| tl.float32)
    else if dtype.isConstOf ``SourceDType.fp16 then `(tritonDType| tl.float16)
    else if dtype.isConstOf ``SourceDType.bf16 then `(tritonDType| tl.bfloat16)
    else if dtype.isConstOf ``SourceDType.fp64 then `(tritonDType| tl.float64)
    else throwError "pointer-element zeros requires supported floating storage"
  return (name.toName,dtype)::(← pairs e.getAppArgs[2]!)
private partial def hasLoad (raw : Syntax) : Bool :=
  let found := match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | _ => Bool.false
  found || raw.getArgs.any hasLoad
private partial def lower (types : List (Name × TSyntax `tritonDType)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower types raw[0]!
  if let `(tritonStmt| $name:ident += $value:tritonExpr) := raw then
    if hasLoad value.raw then
      return ← lower types (← `(tritonStmt| $name:ident = $name:ident + $value:tritonExpr)).raw
  if let `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $key:ident=$target:term)) := raw then
    if let `($name:ident) := target then
      if let .str (.str base "dtype") "element_ty" := name.getId.eraseMacroScopes then
        unless key.getId.eraseMacroScopes==`dtype do Macro.throwError "expected dtype keyword"
        let some (_,dtype) := types.find? (·.1==base) | Macro.throwError "pointer dtype must be known"
        return (← `(tritonExpr| tl.zeros([$dims,*], $dtype:tritonDType))).raw
  return raw.setArgs (← raw.getArgs.mapM (lower types))
@[term_elab IntegerCallArguments.withIntegers]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let types ← pairs (← mkAppM ``CallSignature.pointers #[profile])
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok text => pure text | .error message => throwError "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwError "{message}"
  let `(triton { $body:tritonStmt* }) := (← liftMacroM (lower types parsed)) | throwError "complete body required"
  let source := "{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  let stx := stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))
  ScalarInt64StrideWhile.elaborateInvocation stx expected
end VeriTile.Triton.DSL.PointerDTypeZeros
