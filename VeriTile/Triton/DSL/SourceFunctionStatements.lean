import VeriTile.Triton.DSL.SourceFunction

/-! Checked statement helpers preserve argument evaluation order, local scope,
and simultaneous tuple results. This pass runs before dtype inference. -/
namespace VeriTile.Triton.DSL.SourceFunction
open Lean

syntax (name := sourceHelperDiscardPair)
  ident ", " "_" ", " "_" " = " ident "(" tritonExpr,* ")" : tritonStmt

private structure ExpansionState where
  next : Nat := 0
  used : List Name := []

private abbrev LowerM := StateT ExpansionState MacroM

private partial def fresh (label : String) : LowerM Ident := do
  let state ← get
  let name := Name.mkSimple ("__veritile_helper_" ++ toString state.next ++ "_" ++ label)
  set { state with next := state.next+1 }
  if state.used.contains name then return ← fresh label
  modify fun state => { state with used := name :: state.used }
  return mkIdent name

private partial def substituteConstants (values : List (Name × TSyntax `tritonExpr))
    (raw : Syntax) : MacroM Syntax := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  if let `(tritonExpr| $id:ident) := expression then
    if let some (_,value) := values.find? (fun p => p.1 == id.getId.eraseMacroScopes) then
      return value.raw
  if raw.isIdent && values.any (fun p => p.1.isPrefixOf raw.getId.eraseMacroScopes) then
    Macro.throwErrorAt raw "helper constexpr parameter requires an expression position"
  let args ← raw.getArgs.mapM (substituteConstants values)
  return match raw with
    | .node info kind _ => .node info kind args
    | _ => raw

private def directCall (statement : TSyntax `tritonStmt) :
    Option (Array (Option Ident) × Ident × Array (TSyntax `tritonExpr)) :=
  match statement with
  | `(tritonStmt| $lhs:ident = $fn:ident($arg:tritonExpr)) => some (#[some lhs],fn,#[arg])
  | `(tritonStmt| $lhs:ident = $fn:ident($a:tritonExpr, $b:tritonExpr $[, $rest:tritonExpr]*)) =>
      some (#[some lhs],fn,#[a,b] ++ rest)
  | `(tritonStmt| $a:ident, $b:ident = $fn:ident($args:tritonExpr,*)) =>
      some (#[some a,some b],fn,args.getElems)
  | `(tritonStmt| $a:ident, $b:ident, $c:ident = $fn:ident($args:tritonExpr,*)) =>
      some (#[some a,some b,some c],fn,args.getElems)
  | `(tritonStmt| $a:ident, $b:ident, _ = $fn:ident($args:tritonExpr,*)) =>
      some (#[some a,some b,some (mkIdent `_) ],fn,args.getElems)
  | `(tritonStmt| $a:ident, _, _ = $fn:ident($args:tritonExpr,*)) =>
      some (#[some a,some (mkIdent `_),some (mkIdent `_)],fn,args.getElems)
  | _ => none

mutual
  private partial def lowerStatements (functions : List Checked)
      (statements : Array (TSyntax `tritonStmt)) (stack : List Name) :
      LowerM (Array (TSyntax `tritonStmt)) := do
    let mut output := #[]
    for statement in statements do
      output := output ++ (← lowerStatement functions statement stack)
    return output

  private partial def lowerCall (functions : List Checked) (function : Checked)
      (destinations : Array (Option Ident)) (arguments : Array (TSyntax `tritonExpr))
      (stack : List Name) : LowerM (Array (TSyntax `tritonStmt)) := do
    if stack.contains function.name then
      Macro.throwError "recursive source helpers are unsupported"
    unless arguments.size == function.parameters.length do
      Macro.throwError "source helper argument count differs from its declaration"
    unless destinations.size == function.results.size do
      Macro.throwError "source helper result count differs from its assignment"
    let mut code : Array (TSyntax `tritonStmt) := #[]
    let mut renamed : List (Name × Ident) := []
    let mut constants : List (Name × TSyntax `tritonExpr) := []
    for (parameter,argument) in function.parameters.zip arguments.toList do
      if function.constexpr.contains parameter then
        let `(tritonExpr| $_:num) := argument
          | Macro.throwErrorAt argument "helper constexpr arguments require an integer literal"
        constants := constants ++ [(parameter,argument)]
      else
        let name ← fresh ("arg_" ++ parameter.toString)
        let argument : TSyntax `tritonExpr := ⟨← expand functions argument.raw stack⟩
        code := code.push (← `(tritonStmt| $name:ident = $argument:tritonExpr))
        renamed := renamed ++ [(parameter,name)]
    for statement in function.assignments do
      let `(tritonStmt| $lhs:ident = $_:tritonExpr) := statement
        | Macro.throwError "invalid checked helper assignment"
      let name := lhs.getId.eraseMacroScopes
      unless renamed.any (fun p => p.1 == name) do
        renamed := renamed ++ [(name,← fresh ("local_" ++ name.toString))]
    let mut body := #[]
    for statement in function.assignments do
      let replaced ← substituteConstants constants (substitute renamed statement.raw)
      body := body.push (⟨replaced⟩ : TSyntax `tritonStmt)
    code := code ++ (← lowerStatements functions body (function.name :: stack))
    let mut returns : Array Ident := #[]
    for result in function.results do
      let raw ← substituteConstants constants (substitute renamed result.raw)
      let expression : TSyntax `tritonExpr := ⟨← expand functions raw (function.name :: stack)⟩
      let name ← fresh "result"
      code := code.push (← `(tritonStmt| $name:ident = $expression:tritonExpr))
      returns := returns.push name
    for (destination,result) in destinations.toList.zip returns.toList do
      if let some destination := destination then
        code := code.push (← `(tritonStmt| $destination:ident = $result:ident))
    return code

  private partial def lowerStatement (functions : List Checked)
      (statement : TSyntax `tritonStmt) (stack : List Name) :
      LowerM (Array (TSyntax `tritonStmt)) := do
    if let some (destinations,callee,arguments) := directCall statement then
      if let some function := functions.find? (fun f => f.name == callee.getId.eraseMacroScopes) then
        -- Retain the existing expression path for its previously supported ABI.
        let old := function.assignments.isEmpty && function.results.size == 1 &&
          function.constexpr.isEmpty && arguments.all (fun arg =>
            match arg with | `(tritonExpr| $_:ident) => Bool.true | _ => Bool.false)
        if !old then return ← lowerCall functions function destinations arguments stack
    match statement with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        let condition : TSyntax `tritonExpr := ⟨← expand functions condition.raw stack⟩
        let yes ← lowerStatements functions yes stack
        let no ← lowerStatements functions no stack
        return #[← `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })]
    | `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
        let condition : TSyntax `tritonExpr := ⟨← expand functions condition.raw stack⟩
        let body ← lowerStatements functions body stack
        return #[← `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* })]
    | `(tritonStmt| $_:ident, _, _ = $_:ident($_:tritonExpr,*)) =>
        Macro.throwErrorAt statement "multi-result source helper calls are unsupported"
    | _ => return #[⟨← expand functions statement.raw stack⟩]
end

/-- Fresh names avoid every caller name. Local assignments and all return
expressions are evaluated before assigning any caller tuple destination. -/
def expandStatements (functions : List Checked) (statements : Array (TSyntax `tritonStmt))
    (callerNames : List String) : MacroM (Array (TSyntax `tritonStmt)) := do
  let used := callerNames.map Name.mkSimple ++ statements.toList.flatMap (fun st => namesIn st.raw)
  return (← (lowerStatements functions statements []).run { used }).1

end VeriTile.Triton.DSL.SourceFunction
