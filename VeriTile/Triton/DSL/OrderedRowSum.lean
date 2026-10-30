import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.WelfordSource
import Lean

/-! An invocation may select the existing ordered FP32 sum tree without
inserting a non-Python keyword into the source. This adapter currently supports
axis 1 of a [1, N] tile. It supplies no associativity or floating-point law. -/
namespace VeriTile.Triton.DSL.OrderedRowSum
open Lean

abbrev Bindings := List (String × TSyntax `term)

private partial def expression (bindings : Bindings) (raw : Syntax) :
    StateT (List String) MacroM Syntax := do
  let children ← raw.getArgs.mapM (expression bindings)
  let rebuilt := match raw with
    | .node info kind _ => Syntax.node info kind children
    | _ => raw
  let `(tritonExpr| tl.sum($input:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) := rebuilt
    | return rebuilt
  let `(tritonExpr| $id:ident) := input | return rebuilt
  let name := id.getId.eraseMacroScopes.toString
  let some (_, tree) := bindings.find? (·.1 == name) | return rebuilt
  -- Fail closed for a different axis, dtype override, or keep_dims request.
  -- The source's rank-one result is the reshape of the same ordered scalar.
  unless kwargs.size == 1 do
    Macro.throwErrorAt rebuilt "ordered row sum requires only axis 1"
  let axisValue ← match kwargs[0]! with
    | `(tritonReduceKwarg| $n:num) => pure n.getNat
    | `(tritonReduceKwarg| axis=$n:num) => pure n.getNat
    | _ => Macro.throwErrorAt rebuilt "ordered row sum requires axis 1"
  unless axisValue == 1 do Macro.throwErrorAt rebuilt "ordered row sum requires axis 1"
  modify (name :: ·)
  return (← `(tritonExpr| tl.reshape(
    tl.sum($input, axis=1, tree=$($tree), keep_dims=true), [1]))).raw

def lower (bindings : Bindings) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  if bindings.isEmpty then return body
  let names := bindings.map Prod.fst
  unless names.eraseDups.length == names.length do
    Macro.throwError "ordered row sum bindings must have distinct source operand names"
  let (result, used) ← (body.mapM fun (st : TSyntax `tritonStmt) => do
    return (⟨← expression bindings st.raw⟩ : TSyntax `tritonStmt)).run []
  unless names.all used.contains do
    Macro.throwError "ordered row sum binding does not match a source sum operand"
  return result

end VeriTile.Triton.DSL.OrderedRowSum
