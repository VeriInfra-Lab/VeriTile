import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.PureJitFunctions

/-! Pure unary helpers in plain assignments inside source if branches. Each
call binds its argument and result at that exact statement. Validation reuses
the existing closed single-return helper frontend. Other call sites fail closed. -/
namespace VeriTile.Triton.DSL.ConditionalPureJitFunctions
open Lean Elab Term Meta

syntax (name := leakyBinding) "leaky_relu" " := " term : tritonCallBinding

/-- Builtin-token and ordinary identifier spellings both require a binding. -/
private def call? (raw : Syntax) : Option (Name × Syntax) := do
  let args:=raw.getArgs
  if args.size==4 && args[1]!.isAtom && args[1]!.getAtomVal=="(" then
    let name:=if args[0]!.isIdent then args[0]!.getId.eraseMacroScopes
      else args[0]!.getAtomVal.toName
    return (name,args[2]!)
  if args.size==3 && args[0]!.isAtom then
    let token:=args[0]!.getAtomVal
    if token.endsWith "(" then return ((token.dropEnd 1).toString.toName,args[1]!)
  none

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "helper metadata requires a statically known list"

private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "helper source must be statically available"
  return text

private def argumentNames (profile : Expr) : TermElabM (List Name) := do
  let mut names := []
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      names := names ++ [(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    names := names ++ [(← textValue name).toName]
  return names

private partial def substitute (parameter : Name) (argument : Ident) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes==parameter then argument.raw
  else raw.setArgs (raw.getArgs.map (substitute parameter argument))

private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_jit_control_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_jit_control_"++toString n))

private partial def rejectCalls (functions : List (Name × Name × Syntax)) (raw : Syntax) : TermElabM Unit := do
  if (identifiers raw).any (fun name=>functions.any (fun f=>f.1==name)) ||
      (call? raw).any (fun call=>functions.any (fun f=>f.1==call.1)) then
    throwErrorAt raw "pure JIT calls require a complete plain assignment at the call site"
  for child in raw.getArgs do rejectCalls functions child

private partial def lower (functions : List (Name × Name × Syntax)) (used : List Name)
    (body : Array (TSyntax `tritonStmt)) : StateRefT Nat TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut result := #[]
  for statement in body do
    -- These tuple RHSs read only their distinct own arguments. Splitting
    -- preserves simultaneous-assignment values and keeps each source cast.
    if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* =
        $x:tritonExpr, $y:tritonExpr $[, $rest:tritonExpr]*) := statement then
      let targets := #[a,b] ++ more
      let values := #[x,y] ++ rest
      let selfCast := targets.size==values.size &&
        (targets.toList.map (·.getId.eraseMacroScopes)).eraseDups.length==targets.size &&
        ((targets.zip values).all fun (target,value)=>match value with
          | `(tritonExpr| tl.cast($own:ident, tl.int64)) =>
            own.getId.eraseMacroScopes==target.getId.eraseMacroScopes
          | _ => Bool.false)
      if selfCast then
        for (target,value) in targets.zip values do
          result := result.push (← `(tritonStmt| $target:ident = $value:tritonExpr))
        continue
    if let `(tritonStmt| $target:ident = $rhs:tritonExpr) := statement then
      if let some (callee,argument) := call? rhs.raw then
       if let some (_,parameter,expression) := functions.find? (fun f=>f.1==callee) then
        let argument : TSyntax `tritonExpr := ⟨argument⟩
        rejectCalls functions target.raw
        rejectCalls functions argument.raw
        let input ← fresh used
        let output ← fresh used
        let expression : TSyntax `tritonExpr := ⟨substitute parameter input expression⟩
        result := result ++ #[← `(tritonStmt| $input:ident = $argument:tritonExpr),
          ← `(tritonStmt| $output:ident = $expression:tritonExpr),
          ← `(tritonStmt| $target:ident = $output:ident)]
        continue
    if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) := statement then
      rejectCalls functions condition.raw
      let yes ← lower functions used yes
      let no ← lower functions used no
      result := result.push (← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* }))
      continue
    if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) := statement then
      rejectCalls functions condition.raw
      let yes ← lower functions used yes
      result := result.push (← `(tritonStmt| if $condition:tritonExpr { $yes* }))
      continue
    rejectCalls functions statement.raw
    result := result.push statement
  return result

@[term_elab PureJitFunctions.withFunctions]
def elaborateInvocation : TermElab := fun stx expected => do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body=>pure body | .error message=>throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete helper caller"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let arguments ← argumentNames profile
  let mut functions : List (Name × Name × Syntax) := []
  for binding in stx[8].getSepArgs do
    let (name,definition) ← match binding with
      | `(tritonCallBinding| leaky_relu := $definition:term) => pure (mkIdent `leaky_relu,definition)
      | `(tritonCallBinding| $name:ident := $definition:term) => pure (name,definition)
      | _ => throwError "expected a named pure JIT helper"
    let key:=name.getId.eraseMacroScopes
    unless key.getPrefix.isAnonymous && !["tl","triton","libdevice"].contains key.toString &&
        !arguments.contains key && !functions.any (fun f=>f.1==key) do
      throwErrorAt name "duplicate or captured pure JIT helper name"
    let definition ← elabTermEnsuringType definition (mkConst ``PureJitFunction)
    let parameter ← textValue (← mkAppM ``PureJitFunction.parameter #[definition])
    let source ← textValue (← mkAppM ``PureJitFunction.source #[definition])
    let checked ← match Parser.runParserCategory (← getEnv) `term
        ("triton_function ("++parameter++") {\n"++source++"\n}") with
      | .ok value=>pure value | .error message=>throwError "{message}"
    unless checked.getKind==``PureJitFunctions.pureFunction do throwError "expected a complete helper definition"
    let checked ← try
        withoutErrToSorry <| elabTermEnsuringType checked (mkConst ``PureJitFunction)
      catch error => throwErrorAt binding "{error.toMessageData}"
    let source ← textValue (← mkAppM ``PureJitFunction.source #[checked])
    let source := source.trimAscii.toString
    let expression ← match Parser.runParserCategory (← getEnv) `tritonExpr (source.drop 7).toString with
      | .ok value=>pure value | .error message=>throwError "{message}"
    functions := functions ++ [(key,parameter.toName,expression)]
  let used:=identifiers parsed ++ identifiers stx ++ arguments
  let (statements,_) ← (lower functions used statements).run 0
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let base:=mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
  CallSignature.elaborateInvocation base expected
end VeriTile.Triton.DSL.ConditionalPureJitFunctions
