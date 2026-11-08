import VeriTile.Triton.DSL.TupleScalarCasts

/-! Resolve pointer element-type equality at any nesting depth. Only immutable
caller pointer metadata participates. The selected Boolean is compile-time;
all source operations and both branches remain in the pasted quotation. -/
namespace VeriTile.Triton.DSL.NestedPointerDType
open Lean Elab Term Meta

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "pointer metadata must have static names"

private def pointer? (name : Name) : Option String := do
  let name := name.eraseMacroScopes.toString
  for suffix in [".dtype.element_ty", ".type.element_ty"] do
    if name.endsWith suffix then return (name.dropEnd suffix.length).toString
  none

private partial def immutable (pointer : Name) (raw : Syntax) : MacroM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/=","//="] do
    if let some i := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString == operator) then
      if (args.take i).any (fun a => a.isIdent && a.getId.eraseMacroScopes == pointer) then
        Macro.throwError "queried dtype requires an immutable pointer parameter"
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        b.isIdent && b.getId.eraseMacroScopes == pointer then
      Macro.throwError "loop shadows a queried pointer parameter"
  for child in raw.getArgs do immutable pointer child

private def dtypeNames : List Name :=
  [`tl.float16,`tl.bfloat16,`tl.float32,`tl.float64,`tl.float8e4nv,`tl.float8e5,
   `tl.int8,`tl.int16,`tl.int32,`tl.int64,`tl.uint8,`tl.uint16,`tl.uint32,`tl.uint64,`tl.int1]
private def dtypeCtors : List Name :=
  [``SourceDType.fp16,``SourceDType.bf16,``SourceDType.fp32,``SourceDType.fp64,
   ``SourceDType.f8e4,``SourceDType.f8e5,``SourceDType.i8,``SourceDType.i16,
   ``SourceDType.i32,``SourceDType.i64,``SourceDType.u8,``SourceDType.u16,
   ``SourceDType.u32,``SourceDType.u64,``SourceDType.boolean]

private def dtype? (bindings : List (String × Name)) (e : TSyntax `tritonExpr) : Option Name := do
  let name := if e.raw.isIdent then e.raw.getId.eraseMacroScopes
    else (e.raw.reprint.getD "").trimAscii.toString.toName
  if dtypeNames.contains name then return name
  let pointer ← pointer? name
  bindings.lookup pointer

private partial def predicate? (bindings : List (String × Name)) (raw : Syntax) : Option Bool :=
  match raw with
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
    let a ← dtype? bindings a; let b ← dtype? bindings b; some (a == b)
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
    let a ← dtype? bindings a; let b ← dtype? bindings b; some (a != b)
  | `(tritonExpr| ($inner:tritonExpr)) => predicate? bindings inner.raw
  | _ => none

/-- Splice a statically selected source branch only into statement sequences.
This occurs before branch-local register typing, as in Triton specialization. -/
private partial def lower (bindings : List (String × Name)) (raw : Syntax) : MacroM Syntax := do
  if raw.isIdent && (pointer? raw.getId).isSome then
    Macro.throwError "unsupported or undeclared pointer dtype query"
  let mut args := #[]
  for child in raw.getArgs do
    match child with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if let some value := predicate? bindings condition.raw then
        for statement in (if value then yes else no) do
          args := args.push (← lower bindings statement.raw)
      else args := args.push (← lower bindings child)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if let some value := predicate? bindings condition.raw then
        if value then
          for statement in yes do args := args.push (← lower bindings statement.raw)
      else args := args.push (← lower bindings child)
    | _ => args := args.push (← lower bindings child)
  return raw.setArgs args

private partial def elaborate (stx : Syntax) (expected : Option Expr) : TermElabM Expr := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let entries ← listValues (← mkAppM ``CallSignature.pointers #[profile])
  let mut bindings : List (String × Name) := []
  for pair in entries do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair])
      | throwError "pointer name must be literal"
    let value ← whnf (← mkAppM ``Prod.snd #[pair])
    if let some (_,dtype) := (dtypeCtors.zip dtypeNames).find? (fun (ctor,_) => value.isConstOf ctor) then
      bindings := (name,dtype)::bindings
    else
      -- Case-split only nullary finite metadata, never runtime kernel inputs.
      for id in (collectFVars {} value).fvarIds do
        let .const typeName _ ← whnf (← inferType (mkFVar id)) | continue
        let .inductInfo info ← getConstInfo typeName | continue
        unless info.numParams == 0 && info.numIndices == 0 &&
            !info.ctors.isEmpty && info.ctors.length ≤ 32 do continue
        unless ← info.ctors.allM (fun ctor => do
            let .ctorInfo c ← getConstInfo ctor | return Bool.false
            return c.numFields == 0) do continue
        let goal ← mkFreshExprSyntheticOpaqueMVar (mkConst ``ComputeKernel)
        for branch in ← goal.mvarId!.cases id do
          branch.mvarId.withContext do
            let selected ← exprToSyntax (branch.subst.apply profile)
            branch.mvarId.assign (← elaborate (stx.setArg 1 selected.raw) (some (← branch.mvarId.getType)))
        return ← instantiateMVars goal
      throwError "pointer dtype must be fixed or depend on finite caller metadata"
  let stx ← PythonFullConstructors.prepareSource stx
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed | .error message => throwErrorAt raw "{message}"
  -- Check only pointers whose dtype is actually queried.
  for (name,_) in bindings do
    let rec used (s : Syntax) : Bool :=
      (s.isIdent && pointer? s.getId == some name) || s.getArgs.any used
    if used parsed then liftMacroM (immutable name.toName parsed)
  let `(triton { $statements:tritonStmt* }) := (← liftMacroM (lower bindings parsed))
    | throwError "expected a complete caller body"
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  TupleScalarCasts.elaborateInvocation
    (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := elaborate
end VeriTile.Triton.DSL.NestedPointerDType
