import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.WelfordSource

/-! Caller-selected ordered tree for the transparent Welford helper.
The copied body keeps the original helper arguments. A scalar int64 full
constructor performs its implicit conversion; an immutable constexpr step
may be propagated into range's natural stride. All declarations remain.
The typed index model retains castNatToInt; native representability is a
separate caller obligation, rather than an opaque floating identity. -/
namespace VeriTile.Triton.DSL.WelfordCallMetadata
open Lean Elab Term Meta

syntax (name := bareWelford) ident ", " ident ", " ident " = " "triton_helpers.welford("
  tritonExpr ", " tritonExpr ", " tritonExpr ", " num ")" : tritonStmt
syntax (name := scalarReshape) "tl.reshape(" tritonExpr ", " "(" ")" ")" : tritonExpr
syntax (name := withTree) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " welford_tree " term " in " "triton " RawSourceText.body : term

private partial def identifiers (raw : Syntax) : List Name :=
  (if raw.isIdent then [raw.getId.eraseMacroScopes] else []) ++ raw.getArgs.toList.flatMap identifiers

private def assignedTargets (raw : Syntax) : List Name :=
  match (⟨raw⟩ : TSyntax `tritonStmt) with
    | `(tritonStmt| $id:ident = $_:tritonExpr)
    | `(tritonStmt| $id:ident : tl.constexpr = $_:tritonExpr)
    | `(tritonStmt| $id:ident := $_:tritonExpr)
    | `(tritonStmt| $id:ident += $_:tritonExpr)
    | `(tritonStmt| $id:ident -= $_:tritonExpr)
    | `(tritonStmt| $id:ident *= $_:tritonExpr)
    | `(tritonStmt| $id:ident /= $_:tritonExpr) => [id.getId.eraseMacroScopes]
    | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
      (#[a,b] ++ more).toList.map (·.getId.eraseMacroScopes)
    | `(tritonStmt| $a:ident, $b:ident, $c:ident = triton_helpers.welford(
        $_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:num))
    | `(tritonStmt| $a:ident, $b:ident, $c:ident = triton_helpers.welford_reduce(
        $_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:tritonExpr)) =>
      [a.getId.eraseMacroScopes,b.getId.eraseMacroScopes,c.getId.eraseMacroScopes]
    | `(tritonStmt| $a:ident, $b:ident := $_:tritonExpr, $_:tritonExpr) =>
      [a.getId.eraseMacroScopes,b.getId.eraseMacroScopes]
    | `(tritonStmt| $a:ident, $b:ident = tl.broadcast($_:tritonExpr, $_:tritonExpr))
    | `(tritonStmt| $a:ident, $b:ident := tl.broadcast($_:tritonExpr, $_:tritonExpr))
    | `(tritonStmt| $a:ident, $b:ident = tl.max($_:tritonExpr $[, $_:tritonReduceKwarg]*))
    | `(tritonStmt| $a:ident, $b:ident := tl.max($_:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      [a.getId.eraseMacroScopes,b.getId.eraseMacroScopes]
    | `(tritonStmt| $a:ident, _ = tl.broadcast($_:tritonExpr, $_:tritonExpr))
    | `(tritonStmt| $a:ident, _ := tl.broadcast($_:tritonExpr, $_:tritonExpr)) =>
      [a.getId.eraseMacroScopes]
    | _ => []

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  let changed := assignedTargets raw
  if changed.any names.contains then
    throwErrorAt raw "Welford constexpr dimensions and propagated steps must remain immutable"
  if let first :: counter :: _ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (identifiers counter).any names.contains then
      throwErrorAt counter "a loop cannot shadow a Welford dimension or propagated step"
  for arg in raw.getArgs do rejectWrites names arg

private partial def lower (tree : Ident) (outputs : List Name) (steps : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← lower tree outputs steps raw[0]!
  if let `(tritonExpr| $id:ident[:, None]) := raw then
    if outputs.contains id.getId.eraseMacroScopes then
      return (← `(tritonExpr| tl.reshape($id:ident, [1,1]))).raw
  if let `(tritonExpr| $value:tritonExpr[None, :]) := raw then
    if let some (operand,dtype) := methodCast? value then
      if let `(tritonDType| tl.int64) := dtype then
        let expanded ← `(tritonExpr| $operand:tritonExpr[None, :])
        return (← `(tritonExpr| tl.cast($expanded:tritonExpr, tl.int64))).raw
  if let `(tritonExpr| $value:tritonExpr[:, None]) := raw then
    if let some (operand,dtype) := methodCast? value then
      if let `(tritonDType| tl.int64) := dtype then
        let expanded ← `(tritonExpr| $operand:tritonExpr[:, None])
        return (← `(tritonExpr| tl.cast($expanded:tritonExpr, tl.int64))).raw
  if let `(tritonStmt| $id:ident = tl.broadcast_to($value:tritonExpr, [$dims:tritonExpr,*])) := raw then
    return (← `(tritonStmt| $id:ident, _ = tl.broadcast($value:tritonExpr, tl.zeros([$dims:tritonExpr,*], tl.float32)))).raw
  if let `(tritonExpr| tl.reshape($value:tritonExpr, ())) := raw then
    return (← `(tritonExpr| tl.reshape($value:tritonExpr, []))).raw
  if let `(tritonStmt| $m:ident, $m2:ident, $w:ident = triton_helpers.welford(
      $a:tritonExpr, $b:tritonExpr, $c:tritonExpr, $axisIndex:num)) := raw then
    unless axisIndex.getNat == 1 do Macro.throwError "Welford metadata supports the original single-row axis 1 helper"
    return (← `(tritonStmt| $m:ident, $m2:ident, $w:ident = triton_helpers.welford(
      $a:tritonExpr, $b:tritonExpr, $c:tritonExpr, 1, tree=$($tree:ident)))).raw
  if let `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, dtype=tl.int64)) := raw then
    let value ← `(tritonExpr| tl.cast($value:tritonExpr, tl.int64))
    let full ← `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, dtype=tl.int64))
    return (← `(tritonExpr| tl.cast($full:tritonExpr, tl.int64))).raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $step:ident) { $body:tritonStmt* }) := raw then
    if let some (_,value) := steps.find? (·.1 == step.getId.eraseMacroScopes) then
      let body : Array (TSyntax `tritonStmt) ← body.mapM fun st => return ⟨← lower tree outputs steps st⟩
      return (← `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $(($value:ident : Nat))) { $body* })).raw
  return raw.setArgs (← raw.getArgs.mapM (lower tree outputs steps))

private partial def inlineLiterals (values : List (Name × Nat)) (raw : Syntax) : MacroM Syntax := do
  if raw.isIdent then
    if let some (_,n) := values.find? (·.1 == raw.getId.eraseMacroScopes) then
      let numeral : TSyntax `num := ⟨Syntax.mkNumLit (toString n)⟩
      return (← `(tritonExpr| $numeral:num)).raw
  return raw.setArgs (← raw.getArgs.mapM (inlineLiterals values))

/-- Only established integer expressions contribute signed shape facts.
Floating casts, loads and unsupported expressions do not inherit facts from
integer indices buried inside their syntax. -/
private partial def integerKind (facts : List (Name × Bool)) (e : TSyntax `tritonExpr) : Option Bool :=
  if let some (_,dtype) := methodCast? e then
    match dtype with | `(tritonDType| tl.int64) => some Bool.true | _ => none
  else match e with
  | `(tritonExpr| $id:ident) => (facts.find? (·.1 == id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| $_:num) => some Bool.false
  | `(tritonExpr| tl.program_id($_:num)) | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr)) => some Bool.false
  | `(tritonExpr| tl.cast($_:tritonExpr, tl.int64))
  | `(tritonExpr| tl.full([], $_:tritonExpr, dtype=tl.int64)) => some Bool.true
  | `(tritonExpr| ($a:tritonExpr)) | `(tritonExpr| $a:tritonExpr[:, None])
  | `(tritonExpr| $a:tritonExpr[None, :]) | `(tritonExpr| tl.reshape($a:tritonExpr, ())) => integerKind facts a
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
    return (← integerKind facts a) || (← integerKind facts b)
  | _ => none

private partial def writes (raw : Syntax) : List Name :=
  let here := assignedTargets raw
  let args := raw.getArgs
  let counter := if args.size > 1 && args[0]!.isAtom && args[1]!.isIdent &&
      ["for","tl.for"].contains args[0]!.getAtomVal.trimAscii.toString
    then [args[1]!.getId.eraseMacroScopes] else []
  here ++ counter ++ raw.getArgs.toList.flatMap writes

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "Welford constexpr names must be literal"

@[term_elab withTree] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut constants : List Name := []
  let mut literals : List (Name × Nat) := []
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "Welford constexpr name must be literal"
    constants := constants ++ [name.toName]
    let id := mkIdent name.toName
    let value ← elabTermEnsuringType id (mkConst ``Nat)
    if let .lit (.natVal n) := (← whnf value) then literals := literals ++ [(name.toName,n)]
  let tree ← elabTerm stx[7] none
  let treeType ← inferType tree
  let base := mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[8],stx[9],stx[10]]
  let prepared ← PythonFullConstructors.prepareSource base
  let raw := prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton " ++ raw.getAtomVal) with
    | .ok parsed => pure parsed | .error message => throwErrorAt raw "{message}"
  rejectWrites constants parsed
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete Welford caller"
  let mut outputs : List Name := []
  for stmt in stmts do
    if let `(tritonStmt| $a:ident, $b:ident, $c:ident = triton_helpers.welford(
        $_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:num)) := stmt then
      let names := [a.getId.eraseMacroScopes,b.getId.eraseMacroScopes,c.getId.eraseMacroScopes]
      for other in stmts do
        if other.raw != stmt.raw then rejectWrites names other
      outputs := outputs ++ names
  let mut candidates : List (Name × Expr) := []
  for st in stmts do
    if let `(tritonStmt| $id:ident = tl.full([], $value:ident, dtype=tl.int64)) := st then
      unless constants.contains value.getId.eraseMacroScopes do continue
      if candidates.any (·.1 == id.getId.eraseMacroScopes) then
        throwErrorAt id "duplicate Welford stride definition"
      let count ← elabTermEnsuringType value (mkConst ``Nat)
      for other in stmts do
        if other.raw != st.raw then rejectWrites [id.getId.eraseMacroScopes] other
      candidates := candidates ++ [(id.getId.eraseMacroScopes,count)]
  let used := identifiers parsed ++ identifiers stx
  let rec freezeSteps (todo : List (Name × Expr)) (values : List (Name × Ident))
      (treeName : Ident) : TermElabM Expr := do
    match todo with
    | (name,value)::rest =>
      let fresh := Name.mkSimple ("__veritile_welford_" ++ (← mkFreshId).toString.replace "." "_")
      unless !used.contains fresh do throwError "Welford metadata hygiene collision"
      withLetDecl fresh (mkConst ``Nat) value fun localValue => do
        let result ← freezeSteps rest (values ++ [(name,mkIdent fresh)]) treeName
        mkLetFVars #[localValue] result
    | [] =>
      let mut active : List (Name × Ident) := []
      let mut lowered : Array (TSyntax `tritonStmt) := #[]
      let mut kinds := constants.map (·,Bool.false)
      for stmt in stmts do
        let mut next := stmt.raw
        if let `(tritonStmt| $id:ident = tl.reshape($value:tritonExpr, ())) := stmt then
          if integerKind kinds value == some Bool.true then
            next ← liftMacroM do
              return (← `(tritonStmt| $id:ident = tl.cast(tl.reshape($value:tritonExpr, []), tl.int64))).raw
        lowered := lowered.push ⟨← liftMacroM do lower treeName outputs active (← inlineLiterals literals next)⟩
        let kind := match stmt with
          | `(tritonStmt| $id:ident = $value:tritonExpr) =>
            (integerKind kinds value).map (id.getId.eraseMacroScopes,·)
          | _ => none
        kinds := kinds.filter fun fact => !(writes stmt.raw).contains fact.1
        if let some fact := kind then kinds := fact :: kinds
        if let `(tritonStmt| $id:ident = tl.full([], $_:ident, dtype=tl.int64)) := stmt then
          if let some binding := values.find? (·.1 == id.getId.eraseMacroScopes) then
            active := active ++ [binding]
      let stmts := lowered
      let raw := Syntax.atom .none ("{\n" ++ String.intercalate "\n" (stmts.toList.map (PythonFullConstructors.render ·.raw)) ++ "\n}")
      let delegated := prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size - 1) raw)
      CallSignature.elaborateInvocation delegated expected
  let fresh := Name.mkSimple ("__veritile_welford_" ++ (← mkFreshId).toString.replace "." "_")
  unless !used.contains fresh do throwError "Welford metadata hygiene collision"
  withLetDecl fresh treeType tree fun captured => do
    let result ← freezeSteps candidates [] (mkIdent fresh)
    mkLetFVars #[captured] result
end VeriTile.Triton.DSL.WelfordCallMetadata
