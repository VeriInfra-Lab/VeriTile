import VeriTile.Triton.DSL.PythonBody

/-! Source definitions for pure single-return Triton helpers. They are caller
metadata, outside each pasted kernel body. The raw helper text remains visible;
only explicitly registered names can be expanded. -/
namespace VeriTile.Triton

structure SourceFunction where
  name : String
  parameters : List String
  body : String

namespace DSL.SourceFunction
open Lean

syntax (name := sourceFunctionReturn) "return " tritonExpr : tritonStmt
syntax (name := tritonSourceFunctionCall) (priority := low)
  ident "(" tritonExpr ", " tritonExpr (", " tritonExpr)* ")" : tritonExpr
syntax "triton_function " ident "(" ident,* ")" " in " "triton" tritonPythonBody : term

macro_rules
  | `(triton_function $fn:ident($args:ident,*) in triton $body:tritonPythonBody) => do
      let parameters := args.getElems.map (fun arg => arg.getId.eraseMacroScopes.toString)
      unless parameters.toList.eraseDups.length == parameters.size do
        Macro.throwErrorAt fn "source helper parameters must be distinct"
      let values := parameters.map (fun value => Syntax.mkStrLit value)
      let source := Syntax.mkStrLit (PythonBody.rawText body)
      let functionName := Syntax.mkStrLit fn.getId.eraseMacroScopes.toString
      `(VeriTile.Triton.SourceFunction.mk $functionName [$values,*] $source)

structure Checked where
  name : Name
  parameters : List Name
  result : TSyntax `tritonExpr

private partial def namesIn (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap namesIn

private partial def rejectSplices (raw : Syntax) : MacroM Unit := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  if let `(tritonExpr| $($_:term)) := expression then
    Macro.throwErrorAt raw "Lean antiquotes do not belong in source helpers"
  for arg in raw.getArgs do rejectSplices arg

/-- Every free source name is a formal parameter, a registered helper or a
intrinsic token already parsed by the Triton grammar. An outer Lean value
cannot supply a helper body, including through a namespace-qualified name. -/
def check (environment : Environment) (functionName : String) (parameters : List String)
    (source : String) (registered : List String) : MacroM Checked := do
  if functionName.isEmpty || functionName.contains '.' ||
      ["tl","triton","libdevice"].contains functionName then
    Macro.throwError "source helpers require ordinary Python function names"
  if parameters.any (fun p => ["tl","triton","libdevice","float"].contains p) then
    Macro.throwError "source helper parameters cannot shadow intrinsic names"
  let parsedBody : TSyntax `tritonPythonBody :=
    ⟨Syntax.node SourceInfo.none `tritonPythonBody #[Syntax.atom SourceInfo.none source]⟩
  let statements ← PythonBody.statements environment parsedBody
  unless statements.size == 1 do
    Macro.throwError "source helper currently requires exactly one return expression"
  let `(tritonStmt| return $result:tritonExpr) := statements[0]!
    | Macro.throwError "source helper currently requires exactly one return expression"
  rejectSplices result.raw
  unless parameters.eraseDups.length == parameters.length do
    Macro.throwError "source helper parameters must be distinct"
  unless !parameters.contains functionName do
    Macro.throwError "source helper parameter shadows its function name"
  for name in namesIn result.raw do
    let value := name.toString
    let allowed := parameters.any (fun p => value == p || value.startsWith (p ++ ".")) ||
      registered.contains value
    unless allowed do
      Macro.throwError s!"unbound name in source helper: {name}"
  return ⟨Name.mkSimple functionName, parameters.map Name.mkSimple, result⟩

private partial def substitute (parameters : List (Name × Ident)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes
    match parameters.find? (fun p => p.1.isPrefixOf name) with
    | some (parameter, argument) =>
        mkIdentFrom raw (name.replacePrefix parameter argument.getId.eraseMacroScopes)
    | none => raw
  else match raw with
    | .node info kind args => .node info kind (args.map (substitute parameters))
    | _ => raw

/-- Arguments must already be values named by the source, so expansion neither
duplicates a load/arithmetic expression nor moves its computation precision.
General argument capture and multi-statement helpers require separate support. -/
partial def expand (functions : List Checked) (raw : Syntax)
    (stack : List Name := []) : MacroM Syntax := do
  -- The legacy hand-written DSL has placeholder tuple-call productions.
  -- A pasted Python writer must never inherit their empty-statement fallback.
  let statement : TSyntax `tritonStmt := ⟨raw⟩
  match statement with
  | `(tritonStmt| $_:ident, $_:ident = $_:ident($_:tritonExpr,*))
  | `(tritonStmt| $_:ident, $_:ident, $_:ident = $_:ident($_:tritonExpr,*))
  | `(tritonStmt| $_:ident, $_:ident, _ = $_:ident($_:tritonExpr,*)) =>
      Macro.throwErrorAt raw "multi-result source helper calls are unsupported"
  | _ => pure ()
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let mut call : Option (Ident × Array (TSyntax `tritonExpr)) := match expression with
    | `(tritonExpr| $callee:ident($argument:tritonExpr)) => some (callee, #[argument])
    | `(tritonExpr| $callee:ident($first:tritonExpr, $second:tritonExpr $[, $rest:tritonExpr]*)) =>
        some (callee, #[first,second] ++ rest)
    | _ => none
  -- Some existing shorthand productions reserve a whole token, e.g.
  -- silu(. A registered Python definition still owns that callee's meaning.
  if call.isNone then
    let args := raw.getArgs
    if args.size >= 2 && args[0]!.isAtom then
      let head := args[0]!.getAtomVal
      let opening := head.endsWith "("
      let calleeName := if opening then (head.dropEnd 1).toString else head
      let separated := args.size >= 3 && args[1]!.isAtom && args[1]!.getAtomVal == "("
      if (opening || separated) &&
          functions.any (fun f => f.name.toString == calleeName) then
        let first := if opening then 1 else 2
        let pieces := (args.extract first (args.size-1)).filter (fun arg => !arg.isAtom)
        let arguments ← pieces.mapM fun arg => do
          if arg.isIdent then
            let id : Ident := ⟨arg⟩
            `(tritonExpr| $id:ident)
          else pure (⟨arg⟩ : TSyntax `tritonExpr)
        call := some (mkIdent (Name.mkSimple calleeName),arguments)
  if let some (callee,arguments) := call then
    if let some function := functions.find? (fun f => f.name == callee.getId.eraseMacroScopes) then
      if stack.contains function.name then
        Macro.throwErrorAt callee "recursive source helpers are unsupported"
      unless arguments.size == function.parameters.length do
        Macro.throwErrorAt callee "source helper argument count differs from its declaration"
      let identifiers ← arguments.toList.mapM fun (argument : TSyntax `tritonExpr) => do
        let `(tritonExpr| $id:ident) := argument
          | Macro.throwErrorAt argument "source helper arguments must be named source values"
        if functions.any (fun function => function.name == id.getId.eraseMacroScopes) then
          Macro.throwErrorAt id "source helper arguments must be named source values"
        pure id
      let result := substitute (function.parameters.zip identifiers) function.result.raw
      return ← expand functions result (function.name :: stack)
  if raw.isIdent && functions.any (fun function => function.name == raw.getId.eraseMacroScopes) then
    Macro.throwErrorAt raw "source helpers may only occur in direct calls"
  let args ← raw.getArgs.mapM (fun child => expand functions child stack)
  return match raw with
    | .node info kind _ => Syntax.node info kind args
    | _ => raw

end DSL.SourceFunction
end VeriTile.Triton
