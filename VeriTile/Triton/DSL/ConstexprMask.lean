import VeriTile.Triton.DSL.Syntax

/-! Python None is a compile-time value, never a Boolean tensor register.
This pass specializes immutable Boolean constexpr tests, and tracks None aliases
only through checked control flow. It removes an absent load/store mask, keeps
every tensor operation in order, and rejects uses which would make None a value.
The caller elaborator supplies all Boolean cases; no divisibility assertion or
run-time assumption is introduced here. -/
namespace VeriTile.Triton.DSL.ConstexprMask
open Lean

abbrev Choices := List (Name × Bool)

private partial def boolValue (choices : Choices) (e : TSyntax `tritonExpr) : Option Bool :=
  match e with
  | `(tritonExpr| true) => some Bool.true
  | `(tritonExpr| false) => some Bool.false
  | `(tritonExpr| ($x:tritonExpr)) => boolValue choices x
  | `(tritonExpr| $id:ident) => choices.lookup id.getId.eraseMacroScopes
  | `(tritonExpr| not $x:tritonExpr) => (!·) <$> boolValue choices x
  | `(tritonExpr| $x:tritonExpr and $y:tritonExpr) => do
      return (← boolValue choices x) && (← boolValue choices y)
  | `(tritonExpr| $x:tritonExpr == $y:tritonExpr) => do
      return (← boolValue choices x) == (← boolValue choices y)
  | `(tritonExpr| $x:tritonExpr != $y:tritonExpr) => do
      return (← boolValue choices x) != (← boolValue choices y)
  | _ => none

private partial def noneValue (names : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| None) => Bool.true
  | `(tritonExpr| ($x:tritonExpr)) => noneValue names x
  | `(tritonExpr| $id:ident) => names.contains id.getId.eraseMacroScopes
  | _ => Bool.false

partial def needed (raw : Syntax) : Bool :=
  match raw with
  | `(tritonStmt| $_:ident = $value:tritonExpr) => noneValue [] value || raw.getArgs.any needed
  | `(tritonStmt| $_:ident := $value:tritonExpr) => noneValue [] value || raw.getArgs.any needed
  | _ => raw.getArgs.any needed

private partial def replaceChoices (choices : Choices) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident) := raw then
    if let some value := choices.lookup id.getId.eraseMacroScopes then
      return (← if value then `(tritonExpr| true) else `(tritonExpr| false)).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (replaceChoices choices))
  | _ => return raw

private def maskArgs (names : List Name) (kwargs : Array (TSyntax `tritonMemKwarg))
    (positional : Bool) : MacroM (Array (TSyntax `tritonMemKwarg)) := do
  let mut seen := positional
  let mut result := #[]
  for kw in kwargs do
    if let `(tritonMemKwarg| $id:ident = $value:tritonExpr) := kw then
      if id.getId.eraseMacroScopes == `mask then
        if seen then Macro.throwErrorAt kw "duplicate load/store mask"
        seen := Bool.true
        if noneValue names value then continue
    result := result.push kw
  return result

private partial def expression (names uncertain : List Name) (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
        let kwargs ← maskArgs names kwargs Bool.true
        if noneValue names mask then
          pure (← `(tritonExpr| tl.load($p $[, $kwargs]*))).raw
        else pure (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
        let kwargs ← maskArgs names kwargs Bool.false
        pure (← `(tritonExpr| tl.load($p $[, $kwargs]*))).raw
    | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
        let kwargs ← maskArgs names kwargs Bool.true
        if noneValue names mask then
          if kwargs.isEmpty then pure (← `(tritonStmt| tl.store($p,$v))).raw
          else
            let first := kwargs[0]!
            let rest := kwargs.extract 1 kwargs.size
            pure (← `(tritonStmt| tl.store($p:tritonExpr,$v:tritonExpr,$first:tritonMemKwarg $[, $rest:tritonMemKwarg]*))).raw
        else pure (← `(tritonStmt| tl.store($p:tritonExpr,$v:tritonExpr,$mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
    | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $first:tritonMemKwarg $[, $rest:tritonMemKwarg]*)) => do
        let kwargs ← maskArgs names (#[first] ++ rest) Bool.false
        if kwargs.isEmpty then pure (← `(tritonStmt| tl.store($p,$v))).raw
        else
          let first := kwargs[0]!
          let rest := kwargs.extract 1 kwargs.size
          pure (← `(tritonStmt| tl.store($p:tritonExpr,$v:tritonExpr,$first:tritonMemKwarg $[, $rest:tritonMemKwarg]*))).raw
    | _ => pure raw
  if raw.isIdent && uncertain.contains raw.getId.eraseMacroScopes then
    Macro.throwErrorAt raw "possibly None or undefined mask requires assignment before use"
  if raw.isIdent && names.contains raw.getId.eraseMacroScopes then
    Macro.throwErrorAt raw "None alias is only supported as a direct load/store mask"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (expression names uncertain))
  | _ => return raw

private def sameNames (a b : List Name) : Bool :=
  a.all b.contains && b.all a.contains

/-- Conservative alias closure for None assignments inside a loop. Each
possibly-None name must be defined before reading it on every iteration, and
must be redefined before use after a possibly zero-trip loop. -/
private partial def noneAssignments (known : List Name) (raw : Syntax) : List Name :=
  match raw with
  | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      if noneValue known value then [id.getId.eraseMacroScopes] else []
  | _ => raw.getArgs.toList.flatMap (noneAssignments known)

private partial def nodeCount (raw : Syntax) : Nat :=
  1 + (raw.getArgs.toList.map nodeCount).foldl (· + ·) 0

private def possibleNone (initial : List Name) (body : Array (TSyntax `tritonStmt)) : List Name := Id.run do
  let mut names := initial
  -- Every iteration adds a previously unseen assignment target, or is stable.
  let fuel := body.foldl (fun n st => n + nodeCount st.raw) 1
  for _ in [:fuel] do
    let next := (names ++ body.toList.flatMap (fun st => noneAssignments names st.raw)).eraseDups
    if sameNames names next then return names
    names := next
  return names

private partial def block (choices : Choices) (initial initialUncertain : List Name)
    (stmts : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt) × List Name × List Name) := do
  let mut names := initial
  let mut uncertain := initialUncertain
  let mut output := []
  for st in stmts do
    let branches : Option (TSyntax `tritonExpr × Array (TSyntax `tritonStmt) × Array (TSyntax `tritonStmt) × Bool) :=
      match st with
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => some (c,yes,no,Bool.true)
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) => some (c,yes,#[],Bool.false)
      | _ => none
    if let some (condition,yes,no,hasElse) := branches then
      if let some value := boolValue choices condition then
        let (body,after,afterUncertain) ← block choices names uncertain (if value then yes.toList else no.toList)
        output := output ++ body
        names := after
        uncertain := afterUncertain
      else
        let condition : TSyntax `tritonExpr := ⟨← expression names uncertain condition.raw⟩
        let (yes,afterYes,uncertainYes) ← block choices names uncertain yes.toList
        let (no,afterNo,uncertainNo) ← block choices names uncertain no.toList
        unless sameNames afterYes afterNo do
          Macro.throwErrorAt st "None mask branches require an immutable Boolean constexpr test"
        let yes := yes.toArray
        let no := no.toArray
        output := output ++ [← if hasElse then `(tritonStmt| if $condition { $yes* } else { $no* })
          else `(tritonStmt| if $condition { $yes* })]
        names := afterYes
        uncertain := (uncertainYes ++ uncertainNo).eraseDups
      continue
    let assignment : Option (Ident × TSyntax `tritonExpr) := match st with
      | `(tritonStmt| $id:ident = $value:tritonExpr) => some (id,value)
      | `(tritonStmt| $id:ident := $value:tritonExpr) => some (id,value)
      | _ => none
    if let some (id,value) := assignment then
      let name := id.getId.eraseMacroScopes
      if noneValue names value then names := name :: names.filter (· != name)
      else
        let value : TSyntax `tritonExpr := ⟨← expression names uncertain value.raw⟩
        names := names.filter (· != name)
        output := output ++ [← `(tritonStmt| $id:ident = $value:tritonExpr)]
      uncertain := uncertain.filter (· != name)
      continue
    let loop : Option (Ident × Array (TSyntax `tritonExpr) × Array (TSyntax `tritonStmt)) := match st with
      | `(tritonStmt| for $i:ident in range($a:tritonExpr,$b:tritonExpr,$c:tritonExpr) { $body:tritonStmt* }) => some (i,#[a,b,c],body)
      | `(tritonStmt| for $i:ident in range($a:tritonExpr,$b:tritonExpr) { $body:tritonStmt* }) => some (i,#[a,b],body)
      | `(tritonStmt| for $i:ident in range($a:tritonExpr) { $body:tritonStmt* }) => some (i,#[a],body)
      | _ => none
    if let some (i,args,body) := loop then
      if names.contains i.getId.eraseMacroScopes then
        Macro.throwErrorAt i "loop iterator cannot shadow a None alias"
      let args ← args.mapM fun e => return (⟨← expression names uncertain e.raw⟩ : TSyntax `tritonExpr)
      let possible := possibleNone names body
      let localUncertain := possible.filter (fun name => !names.contains name)
      let (body,after,_) ← block choices names (uncertain ++ localUncertain) body.toList
      unless names.all after.contains do
        Macro.throwErrorAt st "loop cannot change an incoming None alias"
      -- A loop may execute zero times. A local None alias is neither a
      -- definite None nor an old tensor after the loop; require a fresh write.
      uncertain := (uncertain ++ localUncertain).eraseDups
      let body := body.toArray
      let lowered ← match args.toList with
        | [a,b,c] => `(tritonStmt| for $i in range($a,$b,$c) { $body* })
        | [a,b] => `(tritonStmt| for $i in range($a,$b) { $body* })
        | [a] => `(tritonStmt| for $i in range($a) { $body* })
        | _ => Macro.throwUnsupported
      output := output ++ [lowered]
      continue
    -- Other nested statement forms are deliberately rejected if they bind
    -- None; recursively rewriting them would lose scope/control-flow facts.
    if needed st.raw then Macro.throwErrorAt st "unsupported control flow around None mask assignment"
    output := output ++ [⟨← expression names uncertain st.raw⟩]
  return (output,names,uncertain)

def lower (choices : Choices) (stmts : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let (body,_,_) ← block choices [] [] stmts.toList
  body.toArray.mapM fun st => return ⟨← replaceChoices choices st.raw⟩

end VeriTile.Triton.DSL.ConstexprMask
