import Lean.Parser.Extra
import VeriTile.Triton.DSL.Syntax

/-! Optional pointer tests are specialization metadata supplied by the caller.
The quoted Python body retains its None checks. A declared optional pointer
can only occur beneath the corresponding non-None branch; the compiler does
not invent an address or a successful load for an absent argument. -/
namespace VeriTile.Triton.DSL.OptionalPointer
open Lean

/-- Include identifiers in Pratt dispatch without reserving `is` in Lean. -/
meta def optionalIs := Lean.Parser.nonReservedSymbol "is" Bool.true

syntax:50 tritonExpr:51 optionalIs &"not" &"None" : tritonExpr
syntax:50 tritonExpr:51 optionalIs &"None" : tritonExpr

abbrev Presence := List (Name × TSyntax `tritonExpr)

private partial def test? (e : TSyntax `tritonExpr) : Option (Name × Bool) :=
  match e with
  | `(tritonExpr| ($e:tritonExpr)) => test? e
  | `(tritonExpr| $p:ident is not None) => some (p.getId.eraseMacroScopes, Bool.true)
  | `(tritonExpr| $p:ident is None) => some (p.getId.eraseMacroScopes, Bool.false)
  | _ => none

private def value (bindings : Presence) (name : Name) : MacroM (TSyntax `tritonExpr) := do
  let some (_, value) := bindings.find? (fun entry => entry.1 == name)
    | Macro.throwError "None comparison requires a declared optional pointer and caller presence binding"
  return value

private partial def rootedAt (root name : Name) : Bool :=
  root == name || match name with
    | .str parent _ | .num parent _ => rootedAt root parent
    | .anonymous => Bool.false

private partial def expression (bindings : Presence) (active : List Name) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr)) := raw then
    let condition' : TSyntax `tritonExpr := ⟨← expression bindings active condition.raw⟩
    let (yesActive, noActive) := match test? condition with
      | some (pointer, Bool.true) => (pointer :: active, active)
      | some (pointer, Bool.false) => (active, pointer :: active)
      | none => (active, active)
    let yes' : TSyntax `tritonExpr := ⟨← expression bindings yesActive yes.raw⟩
    let no' : TSyntax `tritonExpr := ⟨← expression bindings noActive no.raw⟩
    return (← `(tritonExpr| ($yes' if $condition' else $no'))).raw
  if let some (name, positive) := test? ⟨raw⟩ then
    let present ← value bindings name
    if positive then return present.raw
    else return (← `(tritonExpr| not $present:tritonExpr)).raw
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes
    for (pointer, _) in bindings do
      if rootedAt pointer name && !active.contains pointer then
        Macro.throwErrorAt raw "optional pointer use requires its non-None branch"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (expression bindings active))
  | _ => return raw

private partial def statement (bindings : Presence) (active : List Name)
    (st : TSyntax `tritonStmt) : MacroM (TSyntax `tritonStmt) := do
  match st with
  | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => do
      let condition' : TSyntax `tritonExpr := ⟨← expression bindings active condition.raw⟩
      let (yesActive, noActive) := match test? condition with
        | some (pointer, Bool.true) => (pointer :: active, active)
        | some (pointer, Bool.false) => (active, pointer :: active)
        | none => (active, active)
      let yes ← yes.mapM (statement bindings yesActive)
      let no ← no.mapM (statement bindings noActive)
      `(tritonStmt| if $condition':tritonExpr { $yes* } else { $no* })
  | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) => do
      let condition' : TSyntax `tritonExpr := ⟨← expression bindings active condition.raw⟩
      let yesActive := match test? condition with
        | some (pointer, Bool.true) => pointer :: active
        | _ => active
      let yes ← yes.mapM (statement bindings yesActive)
      `(tritonStmt| if $condition':tritonExpr { $yes* })
  | _ =>
      -- Traverse nested loop bodies with their current presence context.
      let rec visit (raw : Syntax) : MacroM Syntax := do
        -- Parser statement alternatives have distinct node kinds. Recognize
        -- conditional syntax before traversing their children.
        if let `(tritonStmt| if $_:tritonExpr { $_:tritonStmt* }) := raw then
          return (← statement bindings active ⟨raw⟩).raw
        if let `(tritonStmt| if $_:tritonExpr { $_:tritonStmt* } else { $_:tritonStmt* }) := raw then
          return (← statement bindings active ⟨raw⟩).raw
        if let `(tritonExpr| ($_:tritonExpr if $_:tritonExpr else $_:tritonExpr)) := raw then
          return ← expression bindings active raw
        if (test? ⟨raw⟩).isSome then return ← expression bindings active raw
        if raw.isIdent then return ← expression bindings active raw
        match raw with
        | .node info kind args => return .node info kind (← args.mapM visit)
        | _ => return raw
      let raw := st.raw
      match raw with
      | .node info kind args => return ⟨.node info kind (← args.mapM visit)⟩
      | _ => return ⟨← expression bindings active raw⟩

/-- A guarded value expression may depend on surrounding constexpr paths.
Specialize those caller flags before merging source dtype environments. -/
partial def needsConditionalSpecialization (raw : Syntax) : Bool :=
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| ($_ if $condition:tritonExpr else $_)) =>
      (test? condition).isSome || raw.getArgs.any needsConditionalSpecialization
  | _ => raw.getArgs.any needsConditionalSpecialization

def specialize (bindings : Presence) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) :=
  if bindings.isEmpty then pure body else body.mapM (statement bindings [])

end VeriTile.Triton.DSL.OptionalPointer
