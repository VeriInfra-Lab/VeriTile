import VeriTile.Triton.DSL.SourceDType

/-! Compile-time integer aliases in a pasted body. Only declared constexpr
arguments and earlier aliases are accepted. Runtime values are never folded. -/
namespace VeriTile.Triton.DSL.RawBodyConstants
open Lean

private partial def expression (constants : List Name) (value : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match value with
  | `(tritonExpr| $n:num) => `(term| ($n:num : Nat))
  | `(tritonExpr| $name:ident) =>
    unless constants.contains name.getId.eraseMacroScopes do
      Macro.throwErrorAt name "integer constexpr expression uses a runtime or unknown name"
    `(term| ($name:ident : Nat))
  | `(tritonExpr| ($value:tritonExpr)) => expression constants value
  | `(tritonExpr| $left:tritonExpr + $right:tritonExpr) =>
    let left ← expression constants left
    let right ← expression constants right
    `(term| $left + $right)
  | `(tritonExpr| $left:tritonExpr * $right:tritonExpr) =>
    let left ← expression constants left
    let right ← expression constants right
    `(term| $left * $right)
  | `(tritonExpr| $left:tritonExpr // $right:tritonExpr) =>
    let left ← expression constants left
    let right ← expression constants right
    `(term| $left / $right)
  | `(tritonExpr| $left:tritonExpr % $right:tritonExpr) =>
    let left ← expression constants left
    let right ← expression constants right
    `(term| $left % $right)
  | _ => Macro.throwErrorAt value "unsupported integer constexpr expression"

private partial def checkRuntime (bound : List Name) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $_:ident : tl.constexpr = $_:tritonExpr) := raw then
    Macro.throwErrorAt raw "integer constexpr binding under runtime control flow"
  let args := raw.getArgs.toList
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && bound.contains counter.getId.eraseMacroScopes then
      Macro.throwErrorAt counter "runtime loop shadows an integer constexpr"
  for operator in ["=",":=","+=","-=","*=","/="] do
    let args := raw.getArgs.toList
    if let some index := args.findIdx? (fun arg => arg.isAtom && arg.getAtomVal.trimAscii.toString==operator) then
      for name in args.take index do
        if name.isIdent && bound.contains name.getId.eraseMacroScopes then
          Macro.throwErrorAt name "runtime assignment shadows an integer constexpr"
  for arg in raw.getArgs do checkRuntime bound arg

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

private def futureAliases (body : List (TSyntax `tritonStmt)) : List Name :=
  body.filterMap fun stmt => match stmt with
    | `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr) => some name.getId.eraseMacroScopes
    | _ => none

/-- The aliases are Lean lets outside the kernel IR, like the source's
compile-time bindings. The body quotation itself remains unchanged on disk. -/
partial def prepare (constants aliases : List Name) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (Ident × TSyntax `term) × List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return ([],[])
  | stmt :: tail =>
    match stmt with
    | `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr) =>
      let key := name.getId.eraseMacroScopes
      if (constants++aliases).contains key then Macro.throwErrorAt name "duplicate integer constexpr binding"
      let value ← expression (constants++aliases) value
      let (bindings,tail) ← prepare constants (key::aliases) tail
      return ((name,value)::bindings,tail)
    | _ =>
      checkRuntime (constants++aliases) stmt.raw
      if (identifiers stmt.raw).any (futureAliases tail).contains then
        Macro.throwErrorAt stmt "integer constexpr alias used before its binding"
      let (bindings,tail) ← prepare constants aliases tail
      return (bindings,stmt::tail)
end VeriTile.Triton.DSL.RawBodyConstants
