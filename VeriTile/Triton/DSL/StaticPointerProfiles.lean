import VeriTile.Triton.DSL.PythonFullConstructors
import Lean.Util.CollectFVars

/-! Finite pointer dtype profiles are caller metadata. Specialize nullary
enumeration parameters outside the kernel, then delegate each complete pasted
body to the existing call frontend. No source instruction is rewritten here. -/
namespace VeriTile.Triton.DSL.StaticPointerProfiles
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwUnsupportedSyntax

private def choice? (profile : Expr) : TermElabM (Option (LocalDecl × InductiveVal)) := do
  for pair in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
    let value ← whnf (← mkAppM ``Prod.snd #[pair])
    for id in (Lean.collectFVars {} value).fvarIds do
      let declaration ← id.getDecl
      let type ← whnf declaration.type
      let .const name []:=type | continue
      let .inductInfo info ← getConstInfo name | continue
      unless info.numParams==0 && info.numIndices==0 && info.ctors.length>0 &&
          info.ctors.length≤16 && info.levelParams.isEmpty do continue
      let mut finite:=Bool.true
      for constructor in info.ctors do
        let .ctorInfo ctor ← getConstInfo constructor | throwUnsupportedSyntax
        unless ctor.numFields==0 do finite:=Bool.false
      if finite then return some (declaration,info)
  return none

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let some (declaration,info) ← choice? profile | throwUnsupportedSyntax
  let argument:=mkFVar declaration.fvarId
  let mut branches:=#[]
  for constructor in info.ctors do
    let body ← withLetDecl declaration.userName declaration.type (mkConst constructor) fun selected=>do
      let body ← elabTerm stx (some (mkConst ``ComputeKernel))
      mkLetFVars #[selected] body
    branches:=branches.push body
  let motive:=mkLambda `dtype BinderInfo.default declaration.type (mkConst ``ComputeKernel)
  let result ← mkAppOptM (Name.mkStr info.name "casesOn")
    (#[some motive,some argument]++branches.map some)
  ensureHasType expected result
end VeriTile.Triton.DSL.StaticPointerProfiles
