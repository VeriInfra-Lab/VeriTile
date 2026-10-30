import VeriTile.Triton.DSL.PythonConstants

/-! Literal module constants supplied beside the original Python function body.
The source checker connects each entry to its unique module-level
`name = tl.constexpr(literal)` assignment. This compiler accepts only numeric
literals, preserving the existing Python weak-literal promotion; it never
evaluates arbitrary Python or imports an arbitrary Lean value. -/
namespace VeriTile.Triton.DSL.SourceConstant
open Lean

private partial def numericLiteral (expression : TSyntax `tritonExpr) : Bool :=
  match expression with
  | `(tritonExpr| $_:num) | `(tritonExpr| $_:scientific) => Bool.true
  | `(tritonExpr| -$value:tritonExpr) | `(tritonExpr| ($value:tritonExpr)) => numericLiteral value
  | _ => Bool.false

/-- Fail closed on expressions, external names, calls and compiler splices. -/
def check (environment : Environment) (name source : String) : MacroM (Name × TSyntax `tritonExpr) := do
  let chars := name.toList
  unless chars.head?.any (fun c => c.isAlpha || c == '_') &&
      chars.all (fun c => c.isAlphanum || c == '_') &&
      !["tl", "triton", "libdevice", "float", "True", "False", "None"].contains name do
    Macro.throwError "source constants require ordinary unreserved Python names"
  let expression : TSyntax `tritonExpr ← match Parser.runParserCategory environment `tritonExpr source with
    | .ok parsed => pure ⟨parsed⟩
    | .error _ => Macro.throwError "source constant requires a numeric literal"
  unless numericLiteral expression do
    Macro.throwError "source constant requires a numeric literal"
  match PythonConstants.evaluate expression with
  | .error message => Macro.throwError message
  | .ok none => Macro.throwError "source constant requires a numeric literal"
  | .ok (some _) => pure ()
  return (Name.mkSimple name, expression)

end VeriTile.Triton.DSL.SourceConstant
