import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.StatementJitFunctions
import VeriTile.Triton.DSL.ModuleMathConstants
import VeriTile.Triton.DSL.BooleanCallControl

/-! Copied single-return unary helpers may refer to explicitly bound immutable
module constants and other explicitly bound helpers. Calls capture arguments
once and evaluate each return at the source call site. Helper closures cannot
capture caller registers; recursive helper graphs and escaping names fail. -/
namespace VeriTile.Triton
structure ClosedUnaryJitFunction where
  parameter : String
  source : String

namespace DSL.ClosedUnaryJitFunctions
open Lean Elab Term Meta

private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "helper metadata must be static"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "helper metadata must be a static list"
private def simple (name : Name) : Bool :=
  name.isStr && name.getPrefix.isAnonymous &&
    !(["tl","triton","libdevice","None","True","False"] : List String).contains name.toString
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def firstChoice (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then firstChoice raw[0]! else raw.setArgs (raw.getArgs.map firstChoice)
private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let text:=PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok value=>pure value | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete helper/caller body required"
  return body.map (fun stmt=>⟨firstChoice stmt.raw⟩)
private def returnExpression (source : String) : TermElabM Syntax := do
  let body ← parse source
  unless body.size==1 do throwError "closed unary helper requires exactly one return statement"
  let `(tritonStmt| return $results:tritonExpr,*):=body[0]!
    | throwError "closed unary helper requires exactly one return statement"
  unless results.getElems.size==1 do throwError "closed unary helper requires one returned value"
  return results.getElems[0]!.raw

@[term_elab StatementJitFunctions.functionBody]
def elaborateFunction : TermElab := fun stx expected=>do
  let some expectedType:=expected | throwUnsupportedSyntax
  unless ← isDefEq expectedType (mkConst ``ClosedUnaryJitFunction) do throwUnsupportedSyntax
  let parameters:=stx[2].getSepArgs
  unless parameters.size==1 && stx[6].getSepArgs.isEmpty &&
      simple parameters[0]!.getId.eraseMacroScopes do
    throwError "closed unary helper requires one unannotated parameter"
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  discard <| returnExpression source
  ensureHasType expected (mkApp2 (mkConst ``ClosedUnaryJitFunction.mk)
    (mkStrLit parameters[0]!.getId.eraseMacroScopes.toString) (mkStrLit source))

private abbrev Functions := List (Name × Name × Syntax)
private def call? (raw : Syntax) : Option (Name × Syntax) := do
  let args:=raw.getArgs
  if args.size==4 && (args[0]!.isIdent || (args[0]!.isAtom && !args[0]!.getAtomVal.isEmpty)) &&
      args[1]!.isAtom && args[1]!.getAtomVal=="(" then
    let name:=if args[0]!.isIdent then args[0]!.getId.eraseMacroScopes else args[0]!.getAtomVal.toName
    return (name,args[2]!)
  if args.size==3 && args[0]!.isAtom then
    let text:=args[0]!.getAtomVal
    if text.length>1 && text.endsWith "(" then return ((text.dropEnd 1).toString.toName,args[1]!)
  none
private def intrinsics : List String :=
  ["tl.maximum","tl.minimum","tl.extra.cuda.libdevice.tanh","tl.extra.cuda.libdevice.erf"]
private partial def validateExpression (functions : Functions) (constants : List Name)
    (parameter : Name) (stack : List Name) (raw : Syntax) : TermElabM Unit := do
  if raw.isAtom && raw.getAtomVal.contains '$' then throwError "Lean expressions cannot occur in copied helpers"
  if let some (name,argument):=call? raw then
    if let some (_,calleeParameter,body):=functions.find? (·.1==name) then
      if stack.contains name then throwError "recursive JIT helper closures are unsupported"
      validateExpression functions constants parameter stack argument
      validateExpression functions constants calleeParameter (name::stack) body
      return
    unless intrinsics.contains name.toString do throwError "unbound or unsupported helper call: {name}"
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    unless name==parameter || constants.contains name || intrinsics.contains name.toString do
      throwError "JIT helper cannot capture an unbound name: {name}"
  for child in raw.getArgs do validateExpression functions constants parameter stack child
private partial def substitute (parameter : Name) (argument : Ident) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes==parameter then argument.raw
  else raw.setArgs (raw.getArgs.map (substitute parameter argument))
private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_closed_jit_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_closed_jit_"++toString n))
private partial def lowerExpression (functions : Functions) (used : List Name) (raw : Syntax) :
    StateRefT Nat TermElabM (Array (TSyntax `tritonStmt) × Syntax) := do
  if let some (name,argument):=call? raw then
    if let some (_,parameter,body):=functions.find? (·.1==name) then
      let (before,argument) ← lowerExpression functions used argument
      let argument : TSyntax `tritonExpr:=⟨argument⟩
      let input ← fresh used
      let capture ← `(tritonStmt| $input:ident = $argument:tritonExpr)
      let (nested,value) ← lowerExpression functions used (substitute parameter input body)
      let value : TSyntax `tritonExpr:=⟨value⟩
      let output ← fresh used
      return (before++#[capture]++nested++#[← `(tritonStmt| $output:ident = $value:tritonExpr)],
        (← `(tritonExpr| $output:ident)).raw)
  if raw.isIdent && functions.any (·.1==raw.getId.eraseMacroScopes) then
    throwError "JIT helper names cannot escape direct calls"
  let mut before:=#[]
  let mut children:=#[]
  for child in raw.getArgs do
    let (prepended,child) ← lowerExpression functions used child
    before:=before++prepended;children:=children.push child
  return (before,raw.setArgs children)
private def rejectCalls (functions : Functions) (raw : Syntax) : TermElabM Unit := do
  if (ids raw).any (fun name=>functions.any (·.1==name)) then
    throwError "JIT helper calls require a plain assignment at the caller statement"
private partial def lowerStatements (functions : Functions) (used : List Name)
    (body : Array (TSyntax `tritonStmt)) : StateRefT Nat TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut result:=#[]
  for stmt in body do
    if let `(tritonStmt| $target:ident = $value:tritonExpr):=stmt then
      rejectCalls functions target.raw
      let (before,value) ← lowerExpression functions used value
      let value : TSyntax `tritonExpr:=⟨value⟩
      result:=result++before++#[← `(tritonStmt| $target:ident = $value:tritonExpr)]
    else if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }):=stmt then
      rejectCalls functions condition
      let yes ← lowerStatements functions used yes;let no ← lowerStatements functions used no
      result:=result.push (← `(tritonStmt| if $condition { $yes* } else { $no* }))
    else if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }):=stmt then
      rejectCalls functions condition
      let yes ← lowerStatements functions used yes
      result:=result.push (← `(tritonStmt| if $condition { $yes* }))
    else if let `(tritonStmt| for $i:ident in range($start:tritonExpr, 0, -$stride:tritonExpr) { $nested:tritonStmt* }):=stmt then
      for value in [start.raw,stride.raw,i.raw] do rejectCalls functions value
      let nested ← lowerStatements functions used nested
      -- Integer zero and negative integer zero are identical; the latter
      -- spelling exposes the existing signed descending-range parser.
      result:=result.push (← `(tritonStmt| for $i in range($start,-0,-$stride) { $nested* }))
    else if let `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* }):=stmt then
      for value in [start.raw,stop.raw,stride.raw,i.raw] do rejectCalls functions value
      let nested ← lowerStatements functions used nested
      result:=result.push (← `(tritonStmt| for $i in range($start,$stop,$stride) { $nested* }))
    else
      rejectCalls functions stmt
      result:=result.push stmt
  return result

@[term_elab StatementJitFunctions.withFunctions]
def elaborateInvocation : TermElab := fun stx expected=>do
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let body ← parse source
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut parameters : List Name:=[]
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      parameters:=parameters++[(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    parameters:=parameters++[(← textValue value).toName]
  let mut constants : List Name:=[]
  for pair in ← listValues (← mkAppM ``CallSignature.sourceConstants #[profile]) do
    constants:=constants++[(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  let mut functions : Functions:=[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term):=binding | throwError "named JIT helper required"
    let key:=name.getId.eraseMacroScopes
    unless simple key && !parameters.contains key && !constants.contains key && !functions.any (·.1==key) do
      throwError "duplicate or captured JIT helper name"
    let value ← elabTermEnsuringType value (mkConst ``ClosedUnaryJitFunction)
    let parameter:=(← textValue (← mkAppM ``ClosedUnaryJitFunction.parameter #[value])).toName
    let source ← textValue (← mkAppM ``ClosedUnaryJitFunction.source #[value])
    functions:=functions++[(key,parameter,← returnExpression source)]
  unless !functions.isEmpty do throwError "JIT helper bindings cannot be empty"
  for (name,parameter,body) in functions do validateExpression functions constants parameter [name] body
  let used:=ids stx ++ (body.toList.flatMap (ids ·.raw)) ++ parameters ++ constants ++
    (functions.flatMap fun f=>f.1::f.2.1::ids f.2.2)
  let (body,_) ← (lowerStatements functions used body).run 0
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let base:=mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
  let mut booleanConstants:=Bool.false
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let name ← textValue value
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Bool) then booleanConstants:=Bool.true
  if booleanConstants then BooleanCallControl.elaborateInvocation base expected
  else ModuleMathConstants.elaborateInvocation base expected
end DSL.ClosedUnaryJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.ClosedUnaryJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceProcedures.sourceParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``ClosedUnaryJitFunction do throwUnsupportedSyntax
  let parameters := stx[2].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let constants := stx[6].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  unless stx[8].getArgs.isEmpty do
    throwError "symbolic FP32 constants require a SourceProcedure"
  let body ← `(triton_function ($parameters,*) constexpr_args [$constants,*] {})
  elaborateFunction (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.ClosedUnaryJitFunctions
