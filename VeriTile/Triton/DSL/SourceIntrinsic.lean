import VeriTile.Triton.DSL.Expansion.Common

/-! Import bindings live outside the pasted body. The checked binding selects
an intrinsic; a source alias is never interpreted by its spelling alone. -/
namespace VeriTile.Triton.DSL.SourceIntrinsic
open Lean

private def key (i : TSyntax `ident) := i.getId.eraseMacroScopes.toString

private def rejectUse (aliases : List String) (raw : Syntax) : MacroM Unit := do
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes.toString
    if aliases.any (fun a => name == a || name.startsWith (a ++ ".")) then
      Macro.throwErrorAt raw "source intrinsic aliases may only occur in direct calls and cannot be rebound"

/-- Explicit import paths supported by the modeled libdevice intrinsic. -/
def validate (imports : List (String × String)) : MacroM Unit := do
  for (name, target) in imports do
    let .str .anonymous _ := name.toName
      | Macro.throwError "source intrinsic aliases require simple names"
    unless target == "triton.language.extra.libdevice.tanh" ||
        target == "triton.language.extra.cuda.libdevice.tanh" do
      Macro.throwError "unsupported source intrinsic import target"
    if name == "tl" || name == "triton" then
      Macro.throwError "source intrinsic alias captures the language namespace"

partial def lower (imports : List (String × String)) (raw : Syntax) : MacroM Syntax := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let call? := match expression with
    | `(tritonExpr| tanh($argument:tritonExpr)) => some ("tanh", argument)
    | `(tritonExpr| $helperName:ident($argument:tritonExpr)) => some (key helperName, argument)
    | _ => none
  if let some (helperName, argument) := call? then
    if imports.any (fun entry => entry.1 == helperName) then
      let argument : TSyntax `tritonExpr := ⟨← lower imports argument.raw⟩
      return (← `(tritonExpr| tl.extra.cuda.libdevice.tanh($argument:tritonExpr))).raw
    if helperName == "tanh" then
      Macro.throwErrorAt raw "bare tanh requires an explicit checked source import binding"
  rejectUse (imports.map Prod.fst) raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (lower imports))
  | other => return other

end VeriTile.Triton.DSL.SourceIntrinsic
