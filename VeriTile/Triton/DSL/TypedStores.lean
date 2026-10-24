import VeriTile.Triton.DSL.SourceDType

/-! Call-site store conversion with conservative source dtype tracking.
A load and a register copy retain the source pointer's element type. Storing
that value into the same element type is an identity, with no new cast event.
Unknown value types still receive the destination conversion. -/
namespace VeriTile.Triton.DSL.TypedStores
open Lean

abbrev Types := List (Name × Triton.SourceDType)

private def name (id : Ident) : Name := id.getId.eraseMacroScopes

private def lookup (types : Types) (id : Ident) : Option Triton.SourceDType :=
  (types.find? (fun entry => entry.1 == name id)).map Prod.snd

private partial def pointerType (types : Types) (e : TSyntax `tritonExpr) : Option Triton.SourceDType :=
  match e with
  | `(tritonExpr| $id:ident) => lookup types id
  | `(tritonExpr| ($inner:tritonExpr)) => pointerType types inner
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      match pointerType types a, pointerType types b with
      | some dtype, none | none, some dtype => some dtype
      | _, _ => none
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      if (pointerType types b).isSome then none else pointerType types a
  | _ => none

private partial def dtypeOverride (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "dtype") ||
  (raw.isIdent && raw.getId.eraseMacroScopes == `dtype) || raw.getArgs.any dtypeOverride

private partial def valueType (pointers values : Types)
    (e : TSyntax `tritonExpr) : Option Triton.SourceDType :=
  let sameWorkingFloat (a b : TSyntax `tritonExpr) : Option Triton.SourceDType := do
    let dtype ← valueType pointers values a
    if dtype != .fp32 then none
    else if valueType pointers values b == some dtype then some dtype else none
  match e with
  | `(tritonExpr| $id:ident) => lookup values id
  | `(tritonExpr| ($inner:tritonExpr)) => valueType pointers values inner
  | `(tritonExpr| tl.load($p:tritonExpr, $_mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else pointerType pointers p
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else pointerType pointers p
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => sameWorkingFloat a b
  | _ => none

private partial def containsStore (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "tl.store") || raw.getArgs.any containsStore

private partial def rejectPointerRebinding (pointers : Types) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $id:ident = $_value:tritonExpr) := raw then
    if (lookup pointers id).isSome then
      Macro.throwErrorAt raw "pointer alias rebinding inside control flow requires an explicit supported merge"
  if let `(tritonStmt| $id:ident += $_value:tritonExpr) := raw then
    if (lookup pointers id).isSome then
      Macro.throwErrorAt raw "pointer alias rebinding inside control flow requires an explicit supported merge"
  for arg in raw.getArgs do rejectPointerRebinding pointers arg

private def convertedValue (pointers values : Types) (destination value : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr) := do
  let some dtype := pointerType pointers destination
    | Macro.throwErrorAt value "cannot determine tl.store destination dtype from the call signature"
  if valueType pointers values value == some dtype then return value
  let some dt ← dtypeSyntaxOfName? (SourceDType.dtypeName dtype)
    | Macro.throwError "unsupported implicit store dtype"
  if dtype == .boolean then
    `(tritonExpr| (($value:tritonExpr).to(tl.int8)) != $((0 : Int)))
  else
    `(tritonExpr| ($value:tritonExpr).to($dt:tritonDType))

/-- Every store must resolve its destination even when its conversion is an
identity. Unknown control flow fails closed; branch results do not acquire
an unproved source dtype after the merge. -/
partial def lower (pointers : Types) (values : Types)
    (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | stmt :: rest =>
      let mut nextPointers := pointers
      let mut nextValues := values
      let mut stmt := stmt
      match stmt with
      | `(tritonStmt| $id:ident = $value:tritonExpr) =>
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := pointerType pointers value then
            nextPointers := (name id, dtype) :: nextPointers
          if let some dtype := valueType pointers values value then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident += $_value:tritonExpr) =>
          nextValues := values.filter (fun entry => entry.1 != name id)
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          for st in yes ++ no do rejectPointerRebinding pointers st.raw
          let yes := (← lower pointers values yes.toList).toArray
          let no := (← lower pointers values no.toList).toArray
          stmt ← `(tritonStmt| if $condition { $yes* } else { $no* })
          nextValues := []
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          for st in yes do rejectPointerRebinding pointers st.raw
          let yes := (← lower pointers values yes.toList).toArray
          stmt ← `(tritonStmt| if $condition { $yes* })
          nextValues := []
      | _ =>
          rejectPointerRebinding pointers stmt.raw
          if containsStore stmt.raw then
            Macro.throwErrorAt stmt "unsupported store control flow in call signature"
          -- An unsupported statement can assign a tracked value through a
          -- form not understood above. Forget value types conservatively.
          nextValues := []
      return stmt :: (← lower nextPointers nextValues rest)

end VeriTile.Triton.DSL.TypedStores
