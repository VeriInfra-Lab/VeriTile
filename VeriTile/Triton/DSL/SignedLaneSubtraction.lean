import VeriTile.Triton.DSL.StraightLineIdentifiers

/-! Signed subtraction of two broadcast views of a known arange lane vector.
The mathematical address carrier remains natural. Only the lane difference
uses the existing signed32 conversion, so negative differences are retained.
Floating arrays and unknown expressions stay on their ordinary typed path. -/
namespace VeriTile.Triton.DSL.SignedLaneSubtraction
open Lean Elab Term

private def laneView (lanes : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $name:ident[:, None])
  | `(tritonExpr| $name:ident[None, :])=>lanes.contains name.getId.eraseMacroScopes
  | _=>Bool.false

private partial def lower (lanes : List Name) (statements : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt) × Bool) := do
  match statements with
  | []=>return ([],Bool.false)
  | statement::rest=>
    let mut next:=lanes
    let mut statement:=statement
    let mut changed:=Bool.false
    match statement with
    | `(tritonStmt| $name:ident = $value:tritonExpr)=>
      let key:=name.getId.eraseMacroScopes
      next:=lanes.filter (·!=key)
      match value with
      | `(tritonExpr| tl.arange(0, $_:tritonExpr))=>next:=key::next
      | `(tritonExpr| $left:tritonExpr - $right:tritonExpr)=>
        if laneView lanes left && laneView lanes right then
          statement ← `(tritonStmt| $name:ident =
            ($left:tritonExpr).to(tl.int32) - ($right:tritonExpr).to(tl.int32))
          changed:=Bool.true
      | _=>pure ()
    | _=>next:=[]
    let (tail,used) ← lower next rest
    return (statement::tail,changed || used)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body=>pure body | .error message=>throwErrorAt raw "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete source body required"
  let (body,changed) ← liftMacroM (lower [] body.toList)
  unless changed do throwUnsupportedSyntax
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.map (PythonFullConstructors.render ·.raw))++"\n}")
  elabTerm (prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.SignedLaneSubtraction
