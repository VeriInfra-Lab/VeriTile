import VeriTile.Triton.DSL.ClosedBooleanCalls
import VeriTile.Triton.DSL.IntegralBooleanOperands
import VeriTile.Triton.DSL.IntegralAssignments

/-! Python reshape shape tuples and the two-result `tl.split` statement.
The operand is evaluated once before either tuple destination is overwritten.
Only the existing reshape/split operations are emitted. -/
namespace VeriTile.Triton.DSL.PythonTensorTuples
open Lean Elab Term Meta

syntax "tl.reshape(" tritonExpr ", " "(" tritonExpr,* ")" ")" : tritonExpr
syntax (name := splitTuple) ident ", " ident " = " "tl.split(" tritonExpr ")" : tritonStmt

private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def shapes (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM shapes)
  if let `(tritonExpr| tl.reshape($value:tritonExpr, ($dims:tritonExpr,*))):=raw then
    return (← `(tritonExpr| tl.reshape($value:tritonExpr, [$dims,*]))).raw
  return raw

private partial def lower (used : List Name) (next : Nat)
    (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt) × Nat) := do
  let mut out:=#[]
  let mut next:=next
  for statement in body do
    let statement:TSyntax `tritonStmt:=⟨← shapes statement.raw⟩
    match statement with
    | `(tritonStmt| $real:ident, $imag:ident = tl.split($value:tritonExpr))=>
      let freshName (n : Nat):=("__veritile_split_operand_"++toString n).toName
      while used.contains (freshName next) do next:=next+1
      let fresh:=mkIdent (freshName next)
      next:=next+1
      out:=out++#[← `(tritonStmt| $fresh:ident = $value:tritonExpr),
        ← `(tritonStmt| $real:ident = tl.split($fresh:ident, 0)),
        ← `(tritonStmt| $imag:ident = tl.split($fresh:ident, 1))]
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>
      let (yes,n) ← lower used next yes
      let (no,n) ← lower used n no
      next:=n
      out:=out.push (← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* }))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* })=>
      let (yes,n) ← lower used next yes
      next:=n
      out:=out.push (← `(tritonStmt| if $condition:tritonExpr { $yes* }))
    | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* })=>
      let (nested,n) ← lower used next nested
      next:=n
      out:=out.push (← `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested* }))
    | `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr) { $nested:tritonStmt* })=>
      let (nested,n) ← lower used next nested
      next:=n
      out:=out.push (← `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr) { $nested* }))
    | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr) { $nested:tritonStmt* })=>
      let (nested,n) ← lower used next nested
      next:=n
      out:=out.push (← `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr) { $nested* }))
    | _=>out:=out.push statement
  return (out,next)

def prepare (stx : Syntax) : TermElabM Syntax := do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure (first value) | .error message=>throwErrorAt raw "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete tensor tuple source required"
  let (lowered,_) ← liftMacroM (lower (ids parsed) 0 body)
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (lowered.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]!::(← listValues value.getAppArgs[2]!)
  throwError "pointer profile requires static bindings"

private def discriminator (profile : Expr) : TermElabM (Option FVarId) := do
  for entry in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
    let value ← whnf (← mkAppM ``Prod.snd #[entry])
    if value.isConst then continue
    for id in (collectFVars {} value).fvarIds do
      let .const type _ ← whnf (← inferType (mkFVar id)) | continue
      let .inductInfo info ← getConstInfo type | continue
      unless info.numParams==0 && info.numIndices==0 && !info.ctors.isEmpty &&
          info.ctors.length≤32 do continue
      let finite ← info.ctors.allM fun constructor=>do
        let .ctorInfo info ← getConstInfo constructor | return Bool.false
        return info.numFields==0
      if finite then return some id
  return none

partial def elaborateInvocation : TermElab := fun stx expected=>do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  if let some id ← discriminator profile then
    let result ← mkFreshExprSyntheticOpaqueMVar (mkConst ``ComputeKernel)
    for branch in ← result.mvarId!.cases id do
      branch.mvarId.withContext do
        let selected ← exprToSyntax (branch.subst.apply profile)
        let value ← elaborateInvocation (stx.setArg 1 selected.raw) (some (← branch.mvarId.getType))
        branch.mvarId.assign value
    return ← instantiateMVars result
  let stx ← IntegralBooleanOperands.prepare (← prepare stx)
  ClosedBooleanCalls.elaborateInvocation stx expected

end VeriTile.Triton.DSL.PythonTensorTuples
