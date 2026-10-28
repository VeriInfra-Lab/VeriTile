import VeriTile.Triton.DSL.PythonBody

/-! Source definitions for checked straight-line Triton helpers. They are caller
metadata, outside each pasted kernel body. The raw helper text remains visible;
only explicitly registered names can be expanded. Assignment prefixes and
scalar/tuple returns are lowered by `SourceFunctionStatements`; the expression
path here retains support for single-expression helpers with named arguments. -/
namespace VeriTile.Triton

structure SourceFunction where
  name : String
  parameters : List String
  body : String
  /-- Source constexpr parameters, supplied by the caller rather than runtime temporaries. -/
  constexpr : List String := []

namespace DSL.SourceFunction
open Lean

syntax (name := sourceFunctionReturn) "return " tritonExpr : tritonStmt
/-- Python permits an unparenthesized conditional return expression. -/
syntax (name := sourceFunctionConditionalReturn)
  "return " tritonExpr " if " tritonExpr " else " tritonExpr : tritonStmt
syntax (name := tritonSourceFunctionCall) (priority := low)
  ident "(" tritonExpr ", " tritonExpr (", " tritonExpr)* ")" : tritonExpr
syntax (name := sourceFunctionTupleReturn)
  "return " tritonExpr ", " tritonExpr (", " tritonExpr)* : tritonStmt
syntax "triton_function " ident "(" ident,* ")" " in " "triton" tritonPythonBody : term
syntax "triton_function " ident "(" ident,* ")" &"constexpr" "[" ident,* "]" " in " "triton" tritonPythonBody : term

macro_rules
  | `(triton_function $fn:ident($args:ident,*) in triton $body:tritonPythonBody) => do
      let parameters := args.getElems.map (fun arg => arg.getId.eraseMacroScopes.toString)
      unless parameters.toList.eraseDups.length == parameters.size do
        Macro.throwErrorAt fn "source helper parameters must be distinct"
      let values := parameters.map (fun value => Syntax.mkStrLit value)
      let source := Syntax.mkStrLit (PythonBody.rawText body)
      let functionName := Syntax.mkStrLit fn.getId.eraseMacroScopes.toString
      `(VeriTile.Triton.SourceFunction.mk $functionName [$values,*] $source [])

macro_rules
  | `(triton_function $fn:ident($args:ident,*) constexpr [$constants:ident,*] in triton $body:tritonPythonBody) => do
      let base ← `(triton_function $fn:ident($args:ident,*) in triton $body:tritonPythonBody)
      let values := constants.getElems.map (fun arg => Syntax.mkStrLit arg.getId.eraseMacroScopes.toString)
      `(({ $base with constexpr := [$values,*] } : VeriTile.Triton.SourceFunction))

structure Checked where
  name : Name
  parameters : List Name
  result : TSyntax `tritonExpr
  assignments : Array (TSyntax `tritonStmt) := #[]
  results : Array (TSyntax `tritonExpr) := #[]
  constexpr : List Name := []

partial def namesIn (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap namesIn

private partial def rejectSplices (raw : Syntax) : MacroM Unit := do
  if raw.getKind == ``tritonPythonWeakLiteral then
    Macro.throwErrorAt raw "source literal markers are internal compiler syntax"
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  if let `(tritonExpr| $($_:term)) := expression then
    Macro.throwErrorAt raw "Lean antiquotes do not belong in source helpers"
  for arg in raw.getArgs do rejectSplices arg

/-- Every free source name is a formal parameter, a registered helper or a
intrinsic token already parsed by the Triton grammar. An outer Lean value
cannot supply a helper body, including through a namespace-qualified name. -/
def check (environment : Environment) (functionName : String) (parameters : List String)
    (source : String) (registered : List String) (constants : List String := []) : MacroM Checked := do
  if functionName.isEmpty || functionName.contains '.' ||
      ["tl","triton","libdevice"].contains functionName then
    Macro.throwError "source helpers require ordinary Python function names"
  if parameters.any (fun p => ["tl","triton","libdevice","float"].contains p) then
    Macro.throwError "source helper parameters cannot shadow intrinsic names"
  if parameters.any (fun p => p.isEmpty || p.contains '.') then
    Macro.throwError "source helper parameters require ordinary Python names"
  unless parameters.eraseDups.length == parameters.length do
    Macro.throwError "source helper parameters must be distinct"
  unless !parameters.contains functionName do
    Macro.throwError "source helper parameter shadows its function name"
  if parameters.any registered.contains then
    Macro.throwError "source helper parameter shadows a registered helper"
  unless constants.eraseDups.length == constants.length && constants.all parameters.contains do
    Macro.throwError "source helper constexpr names must be distinct parameters"
  let parsedBody : TSyntax `tritonPythonBody :=
    ⟨Syntax.node SourceInfo.none `tritonPythonBody #[Syntax.atom SourceInfo.none source]⟩
  let statements ← PythonBody.statements environment parsedBody
  if statements.isEmpty then Macro.throwError "source helper requires a final return"
  let last := statements[statements.size-1]!
  let results ← match last with
    | `(tritonStmt| return $yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr) => do
        pure #[← `(tritonExpr| ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr))]
    | `(tritonStmt| return $first:tritonExpr, $second:tritonExpr $[, $rest:tritonExpr]*) =>
        pure (#[first,second] ++ rest)
    | `(tritonStmt| return $result:tritonExpr) => pure #[result]
    | _ => Macro.throwError "source helper requires a final return"
  let assignments := statements.pop
  let mut bound := parameters
  for statement in assignments do
    rejectSplices statement.raw
    let `(tritonStmt| $lhs:ident = $rhs:tritonExpr) := statement
      | Macro.throwError "source helper supports ordinary assignments only"
    let name := lhs.getId.eraseMacroScopes.toString
    if name.contains '.' || ["tl","triton","libdevice","float"].contains name || registered.contains name then
      Macro.throwErrorAt lhs "source helper local cannot shadow an intrinsic or helper"
    if constants.contains name then
      Macro.throwErrorAt lhs "source helper constexpr parameter cannot be rebound"
    for used in namesIn rhs.raw do
      let value := used.toString
      unless bound.any (fun p => value == p || value.startsWith (p ++ ".")) || registered.contains value do
        Macro.throwError s!"unbound name in source helper: {used}"
    if !bound.contains name then bound := bound ++ [name]
  for result in results do
    rejectSplices result.raw
    for name in namesIn result.raw do
      let value := name.toString
      unless bound.any (fun p => value == p || value.startsWith (p ++ ".")) || registered.contains value do
        Macro.throwError s!"unbound name in source helper: {name}"
  return {
    name := Name.mkSimple functionName
    parameters := parameters.map Name.mkSimple
    result := results[0]!
    assignments := assignments
    results := results
    constexpr := constants.map Name.mkSimple
  }

partial def substitute (parameters : List (Name × Ident)) (raw : Syntax) : Syntax :=
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
General argument capture and assignment-prefix helpers use
`SourceFunctionStatements.expandStatements` instead. -/
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
      unless function.assignments.isEmpty && function.results.size == 1 && function.constexpr.isEmpty do
        Macro.throwErrorAt callee "statement source helper requires a direct assignment call"
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
