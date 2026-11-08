import VeriTile.Triton.DSL.Syntax
import Lean

/-! Checked module-level bindings outside pasted function bodies. Constant
bindings admit only literal integers. Unary imports name the actual modeled
libdevice target; no alias is interpreted from its spelling alone. Python
source correspondence checks both kinds against their defining module. -/
namespace VeriTile.Triton.DSL.SourceBindings
open Lean

private partial def integerLiteral (expression : TSyntax `tritonExpr) : Bool :=
  match expression with
  | `(tritonExpr| $_:num) => Bool.true
  | `(tritonExpr| -$value:tritonExpr) | `(tritonExpr| ($value:tritonExpr)) => integerLiteral value
  | _ => Bool.false

def constant (environment : Environment) (name source : String) : MacroM (Name × TSyntax `tritonExpr) := do
  let chars := name.toList
  unless chars.head?.any (fun c => c.isAlpha || c == '_') &&
      chars.all (fun c => c.isAlphanum || c == '_') &&
      !["tl", "triton", "libdevice", "float", "True", "False", "None"].contains name do
    Macro.throwError "source constants require ordinary unreserved Python names"
  let expression : TSyntax `tritonExpr ← match Parser.runParserCategory environment `tritonExpr source with
    | .ok parsed => pure ⟨parsed⟩
    | .error _ => Macro.throwError "source constant requires an integer literal"
  unless integerLiteral expression do
    Macro.throwError "source constant requires an integer literal"
  return (Name.mkSimple name,expression)

def validateUnary (imports : List (String × String)) : MacroM Unit := do
  for (name,target) in imports do
    let .str .anonymous _ := name.toName
      | Macro.throwError "source intrinsic aliases require simple names"
    unless target == "triton.language.extra.libdevice.tanh" ||
        target == "triton.language.extra.cuda.libdevice.tanh" do
      Macro.throwError "unsupported source intrinsic import target"
    if ["tl","triton","libdevice","float"].contains name then
      Macro.throwError "source intrinsic alias captures a language namespace"

private def rejectAlias (imports : List (String × String)) (raw : Syntax) : MacroM Unit := do
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes.toString
    if imports.any (fun entry => name == entry.1 || name.startsWith (entry.1 ++ ".")) then
      Macro.throwErrorAt raw "source intrinsic aliases may only occur in direct calls and cannot be rebound"

partial def lowerUnary (imports : List (String × String)) (raw : Syntax) : MacroM Syntax := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let call? := match expression with
    | `(tritonExpr| tanh($argument:tritonExpr)) => some ("tanh",argument)
    | `(tritonExpr| $helper:ident($argument:tritonExpr)) =>
        some (helper.getId.eraseMacroScopes.toString,argument)
    | _ => none
  if let some (name,argument) := call? then
    if imports.any (fun entry => entry.1 == name) then
      let argument : TSyntax `tritonExpr := ⟨← lowerUnary imports argument.raw⟩
      return (← `(tritonExpr| tl.extra.cuda.libdevice.tanh($argument:tritonExpr))).raw
  rejectAlias imports raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (lowerUnary imports))
  | other => return other

end VeriTile.Triton.DSL.SourceBindings
