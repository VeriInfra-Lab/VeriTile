import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.Core.Types

/-! Pointer element dtypes are compile-time metadata. Specialize a source
dtype test before elaborating the branch-local register types. The complete
source quotation remains the input; every selected branch retains its suffix. -/
namespace VeriTile.Triton.DSL.RawBodyDTypeControl
open Lean

private def dtypeTerm? (bindings : List (String × TSyntax `term))
    (e : TSyntax `tritonExpr) : MacroM (Option (TSyntax `term)) := do
  match e with
  | `(tritonExpr| tl.float16) => return some (← `(FloatDType.fp16))
  | `(tritonExpr| tl.bfloat16) => return some (← `(FloatDType.bf16))
  | `(tritonExpr| tl.float32) => return some (← `(FloatDType.fp32))
  | `(tritonExpr| tl.float64) => return some (← `(FloatDType.real))
  | `(tritonExpr| tl.float8e4nv) => return some (← `(FloatDType.f8e4))
  | `(tritonExpr| tl.float8e5) => return some (← `(FloatDType.f8e5))
  | _ => pure ()
  let `(tritonExpr| $id:ident) := e | return none
  let name := id.getId.eraseMacroScopes.toString
  for suffix in [".dtype.element_ty", ".type.element_ty"] do
    if name.endsWith suffix then
      let pointer := (name.dropEnd suffix.length).toString
      return (bindings.find? (·.1 == pointer)).map (·.2)
  match name with
  | "tl.float16" => return some (← `(FloatDType.fp16))
  | "tl.bfloat16" => return some (← `(FloatDType.bf16))
  | "tl.float32" => return some (← `(FloatDType.fp32))
  | "tl.float64" => return some (← `(FloatDType.real))
  | "tl.float8e4nv" => return some (← `(FloatDType.f8e4))
  | "tl.float8e5" => return some (← `(FloatDType.f8e5))
  | _ => return none

private partial def condition? (bindings : List (String × TSyntax `term))
    (e : TSyntax `tritonExpr) : MacroM (Option (TSyntax `term)) := do
  match e with
  | `(tritonExpr| ($inner:tritonExpr)) => condition? bindings inner
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) =>
    let some a ← dtypeTerm? bindings a | return none
    let some b ← dtypeTerm? bindings b | return none
    return some (← `(($a : FloatDType) = ($b : FloatDType)))
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) =>
    let some a ← dtypeTerm? bindings a | return none
    let some b ← dtypeTerm? bindings b | return none
    return some (← `(($a : FloatDType) ≠ ($b : FloatDType)))
  | _ => return none

private def split? (bindings : List (String × TSyntax `term)) :
    List (TSyntax `tritonStmt) → MacroM (Option
      (TSyntax `term × List (TSyntax `tritonStmt) × List (TSyntax `tritonStmt)))
  | [] => pure none
  | st :: rest => do
    match st with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if let some condition ← condition? bindings condition then
        return some (condition,yes.toList ++ rest,no.toList ++ rest)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if let some condition ← condition? bindings condition then
        return some (condition,yes.toList ++ rest,rest)
    | _ => pure ()
    match ← split? bindings rest with
    | none => return none
    | some (condition,yes,no) => return some (condition,st :: yes,st :: no)

/-- A static choice returns a complete typed kernel in either case. It never
turns a runtime data comparison into a compile-time choice. -/
partial def emit (bindings : List (String × TSyntax `term))
    (leaf : List (TSyntax `tritonStmt) → MacroM (TSyntax `term))
    (body : List (TSyntax `tritonStmt)) : MacroM (TSyntax `term) := do
  match ← split? bindings body with
  | none => leaf body
  | some (condition,yes,no) =>
    let yes ← emit bindings leaf yes
    let no ← emit bindings leaf no
    `(if $condition then $yes else $no)

end VeriTile.Triton.DSL.RawBodyDTypeControl
