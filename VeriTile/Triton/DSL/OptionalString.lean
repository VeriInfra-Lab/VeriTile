import Lean.Elab.Term
import VeriTile.Triton.DSL.OptionalPointer

/-! Optional string constexpr arguments retain their Python None and string
comparisons. Only finite literal caller choices are supported. Branch facts
come from those actual choices, not an invented default value or a numerical
assumption. -/
namespace VeriTile.Triton.DSL.OptionalString
open Lean Elab Term Meta

private partial def strings (value : Expr) : TermElabM (List String) := do
  let value ← instantiateMVars value
  if let .lit (.strVal s) := value then return [s]
  if let .fvar id := value then
    if let some rhs := (← id.getDecl).value? then return ← strings rhs
  if let .letE _ _ rhs body _ := value then return ← strings (body.instantiate1 rhs)
  if let .mdata _ inner := value then return ← strings inner
  if value.isAppOfArity ``ite 5 then
    let args := value.getAppArgs
    return ((← strings args[3]!) ++ (← strings args[4]!)).eraseDups
  let reduced ← whnf value
  if let .lit (.strVal s) := reduced then return [s]
  throwError "optional string constexpr requires finite literal caller choices"

private partial def values (value : Expr) : TermElabM (List (Option String)) := do
  let value ← instantiateMVars value
  if value.isAppOfArity ``Option.none 1 then return [none]
  if value.isAppOfArity ``Option.some 2 then
    return (← strings value.getAppArgs[1]!).map some
  if let .fvar id := value then
    if let some rhs := (← id.getDecl).value? then return ← values rhs
  if let .letE _ _ rhs body _ := value then return ← values (body.instantiate1 rhs)
  if let .mdata _ inner := value then return ← values inner
  if value.isAppOfArity ``ite 5 then
    let args := value.getAppArgs
    return ((← values args[3]!) ++ (← values args[4]!)).eraseDups
  let reduced ← whnf value
  if reduced.isAppOfArity ``Option.none 1 then return [none]
  if reduced.isAppOfArity ``Option.some 2 then
    return (← strings reduced.getAppArgs[1]!).map some
  throwError "optional string constexpr requires finite literal caller choices"

private abbrev Choices := List (Name × List (Option String))
private abbrev Test := Name × Option String × Bool

private partial def test? (e : TSyntax `tritonExpr) : Option Test :=
  match e with
  | `(tritonExpr| ($e:tritonExpr)) => test? e
  | `(tritonExpr| $id:ident is None) => some (id.getId.eraseMacroScopes,none,Bool.true)
  | `(tritonExpr| $id:ident is not None) => some (id.getId.eraseMacroScopes,none,Bool.false)
  | `(tritonExpr| $id:ident == $s:str) => some (id.getId.eraseMacroScopes,some s.getString,Bool.true)
  | `(tritonExpr| $id:ident != $s:str) => some (id.getId.eraseMacroScopes,some s.getString,Bool.false)
  | _ => none

private def condition (test : Test) : MacroM (TSyntax `tritonExpr) := do
  let (name,value,equal) := test
  let id : TSyntax `term := ⟨mkIdent name⟩
  let rhs : TSyntax `term ← match value with
    | none => `((none : Option String))
    | some s => do
      let lit : TSyntax `term := ⟨Syntax.mkStrLit s⟩
      `(some $lit)
  let bool : TSyntax `term ← if equal then `($id == $rhs) else `($id != $rhs)
  `(tritonExpr| $(($bool:term : Bool)))

private partial def expression (choices : Choices) (raw : Syntax) : MacroM Syntax := do
  if let some test := test? ⟨raw⟩ then
    if choices.any (fun p => p.1 == test.1) then return (← condition test).raw
  if raw.isIdent && choices.any (fun p => p.1 == raw.getId.eraseMacroScopes) then
    Macro.throwErrorAt raw "optional string constexpr may only occur in None/string comparisons"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (expression choices))
  | _ => return raw

private partial def block (choices : Choices) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  let mut result := []
  for stmt in body do
    let branches : Option (TSyntax `tritonExpr × Array (TSyntax `tritonStmt) ×
        Array (TSyntax `tritonStmt) × Bool) := match stmt with
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => some (c,yes,no,Bool.true)
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) => some (c,yes,#[],Bool.false)
      | _ => none
    if let some (c,yes,no,hasElse) := branches then
      let mut yesChoices := choices
      let mut noChoices := choices
      if let some (name,literal,equal) := test? c then
        if let some (_,allowed) := choices.find? (fun p => p.1 == name) then
          let yesValues := allowed.filter (fun v => (v == literal) == equal)
          let noValues := allowed.filter (fun v => (v == literal) != equal)
          yesChoices := choices.map (fun p => if p.1 == name then (name,yesValues) else p)
          noChoices := choices.map (fun p => if p.1 == name then (name,noValues) else p)
          if yesValues.isEmpty then
            result := result ++ (← block noChoices no.toList)
            continue
          if noValues.isEmpty then
            result := result ++ (← block yesChoices yes.toList)
            continue
      let c : TSyntax `tritonExpr := ⟨← expression choices c.raw⟩
      let yes := (← block yesChoices yes.toList).toArray
      let no := (← block noChoices no.toList).toArray
      result := result ++ [← if hasElse then `(tritonStmt| if $c { $yes* } else { $no* })
        else `(tritonStmt| if $c { $yes* })]
      continue
    match stmt with
    | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← block choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i:ident in range($a:tritonExpr,$b:tritonExpr,$c:tritonExpr) { $body* })]
    | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← block choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i:ident in range($a:tritonExpr,$b:tritonExpr) { $body* })]
    | `(tritonStmt| for $i:ident in range($a:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← block choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i:ident in range($a:tritonExpr) { $body* })]
    | _ => result := result ++ [⟨← expression choices stmt.raw⟩]
  return result

def specialize (constants : List String) (body : Array (TSyntax `tritonStmt)) :
    TermElabM (Array (TSyntax `tritonStmt) × List Name) := do
  let mut choices : Choices := []
  for constant in constants do
    let name := Name.mkSimple constant
    let value ← elabTerm (mkIdent name) none
    let type ← whnf (← inferType value)
    if type.isAppOfArity ``Option 1 && type.getAppArgs[0]!.isConstOf ``String then
      choices := choices ++ [(name, ← values value)]
  if choices.isEmpty then return (body,[])
  return ((← liftMacroM <| block choices body.toList).toArray,choices.map Prod.fst)
end VeriTile.Triton.DSL.OptionalString
