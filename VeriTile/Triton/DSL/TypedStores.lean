import VeriTile.Triton.DSL.SourceDType

/-! Call-site store conversion with conservative source dtype tracking.
A load and a register copy retain the source pointer's element type. Storing
that value into the same element type is an identity, with no new cast event.
Unknown value types still receive the destination conversion. -/
namespace VeriTile.Triton.DSL.TypedStores
open Lean

abbrev Types := List (Name × Triton.SourceDType)

inductive ValueInfo where
  | typed (dtype : Triton.SourceDType)
  | integerLiteral | floatingLiteral
  deriving BEq
abbrev Values := List (Name × ValueInfo)

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

private def sourceKinds : List Triton.SourceDType :=
  [.fp16,.bf16,.fp32,.fp64,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean]

private partial def dtypeToken (raw : Syntax) : Option Name :=
  if raw.isAtom then
    (sourceKinds.find? (fun dtype => (SourceDType.dtypeName dtype).toString == raw.getAtomVal)).map SourceDType.dtypeName
  else if raw.isIdent then some raw.getId.eraseMacroScopes
  else if raw.getNumArgs == 1 then dtypeToken raw[0]
  else none

private def explicitCastType (e : TSyntax `tritonExpr) : Option Triton.SourceDType := do
  let direct : Option (TSyntax `tritonDType) := match e with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | _ => none
  if let some dt := direct then
    let target ← dtypeToken dt.raw
    return ← sourceKinds.find? (fun dtype => SourceDType.dtypeName dtype == target)
  unless (methodCast? e).isSome || e.raw.getArgs.any
      (fun arg => arg.isAtom && arg.getAtomVal == "to") do failure
  let target ← (methodCastTargetName? e).orElse fun _ => do
    let (_, dtype) ← methodCast? e
    dtypeToken dtype.raw
  sourceKinds.find?
    (fun dtype => SourceDType.dtypeName dtype == target)

private partial def valueType (pointers : Types) (values : Values)
    (e : TSyntax `tritonExpr) : Option ValueInfo :=
  let sameWorkingFloat (a b : TSyntax `tritonExpr) : Option Triton.SourceDType := do
    let left ← valueType pointers values a
    let right ← valueType pointers values b
    match left, right with
    | .typed .fp32, .typed .fp32
    | .typed .fp32, .integerLiteral | .integerLiteral, .typed .fp32
    | .typed .fp32, .floatingLiteral | .floatingLiteral, .typed .fp32 => some .fp32
    | _, _ => none
  if let some dtype := explicitCastType e then some (.typed dtype) else
  match e with
  | `(tritonExpr| $id:ident) => (values.find? (fun entry => entry.1 == name id)).map Prod.snd
  | `(tritonExpr| ($inner:tritonExpr)) => valueType pointers values inner
  | `(tritonExpr| $_:num) | `(tritonExpr| -$_:num) => some .integerLiteral
  | `(tritonExpr| $_:scientific) | `(tritonExpr| -$_:scientific)
  | `(tritonExpr| float($_:term)) | `(tritonExpr| -float($_:term)) => some .floatingLiteral
  | `(tritonExpr| tl.load($p:tritonExpr, $_mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else (pointerType pointers p).map ValueInfo.typed
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else (pointerType pointers p).map ValueInfo.typed
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| tl.where($_:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
      (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| tl.sin($a:tritonExpr)) | `(tritonExpr| tl.math.sin($a:tritonExpr))
  | `(tritonExpr| tl.cos($a:tritonExpr)) | `(tritonExpr| tl.exp($a:tritonExpr))
  | `(tritonExpr| tl.sigmoid($a:tritonExpr)) | `(tritonExpr| tl.rsqrt($a:tritonExpr))
  | `(tritonExpr| tl.log($a:tritonExpr)) | `(tritonExpr| tl.sqrt($a:tritonExpr)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.sum($a:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| -$a:tritonExpr) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.trans($a:tritonExpr)) => valueType pointers values a
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

private def convertedValue (pointers : Types) (values : Values) (destination value : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr) := do
  let some dtype := pointerType pointers destination
    | Macro.throwErrorAt value "cannot determine tl.store destination dtype from the call signature"
  if valueType pointers values value == some (.typed dtype) then return value
  let some dt ← dtypeSyntaxOfName? (SourceDType.dtypeName dtype)
    | Macro.throwError "unsupported implicit store dtype"
  if dtype == .boolean then
    `(tritonExpr| (($value:tritonExpr).to(tl.int8)) != $((0 : Int)))
  else
    `(tritonExpr| ($value:tritonExpr).to($dt:tritonDType))

/-- Every store must resolve its destination even when its conversion is an
identity. Unknown control flow fails closed; branch results do not acquire
an unproved source dtype after the merge. -/
partial def lower (pointers : Types) (values : Values)
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
