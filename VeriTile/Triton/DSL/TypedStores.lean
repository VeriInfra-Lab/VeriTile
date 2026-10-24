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
  | integerLiteral | integerScalar | floatingLiteral
  deriving BEq
abbrev Values := List (Name × ValueInfo)

/-- Internal provenance marker for a supplied FP32 literal. The call frontend
checks every occurrence's actual computation precision after elaboration. -/
@[irreducible] def fp32CallValue (value : ℝ) : ℝ := value

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

private def constructorType (e : TSyntax `tritonExpr) : Option Triton.SourceDType := do
  let args := e.raw.getArgs
  let first ← args[0]?
  unless first.isAtom && ["tl.zeros(","tl.full("].contains first.getAtomVal do failure
  for i in [:args.size] do
    if args[i]!.isIdent && args[i]!.getId.eraseMacroScopes == `dtype && i+2 < args.size then
      let target ← dtypeToken args[i+2]!
      return ← sourceKinds.find? (fun dtype => SourceDType.dtypeName dtype == target)
  for arg in args do
    if let some target := dtypeToken arg then
      if let some dtype := sourceKinds.find? (fun dtype => SourceDType.dtypeName dtype == target) then
        return dtype
  match e with
  | `(tritonExpr| tl.zeros([$_:tritonExpr,*])) => return .fp32
  | _ => failure

private partial def valueType (pointers : Types) (values : Values)
    (e : TSyntax `tritonExpr) : Option ValueInfo :=
  let sameWorkingFloat (a b : TSyntax `tritonExpr) : Option Triton.SourceDType := do
    let left ← valueType pointers values a
    let right ← valueType pointers values b
    match left, right with
    | .typed .fp32, .typed .fp32
    | .typed .fp32, .integerLiteral | .integerLiteral, .typed .fp32
    | .typed .fp32, .integerScalar | .integerScalar, .typed .fp32
    | .typed .fp32, .floatingLiteral | .floatingLiteral, .typed .fp32 => some .fp32
    | .typed .fp32, .typed integer | .typed integer, .typed .fp32 =>
        if [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains integer then some .fp32 else none
    | _, _ => none
  if let some dtype := (explicitCastType e).orElse (fun _ => constructorType e) then some (.typed dtype) else
  match e with
  | `(tritonExpr| $id:ident) => (values.find? (fun entry => entry.1 == name id)).map Prod.snd
  | `(tritonExpr| ($inner:tritonExpr)) => valueType pointers values inner
  | `(tritonExpr| $((VeriTile.Triton.DSL.TypedStores.fp32CallValue $_:term : ℝ))) => some (.typed .fp32)
  | `(tritonExpr| $(($_:term : Nat))) | `(tritonExpr| $(($_:term : Int))) => some .integerScalar
  | `(tritonExpr| $(($_:term : ℝ))) => some .floatingLiteral
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
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr))
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
      (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| $_:tritonExpr < $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr <= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr > $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr >= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr == $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr != $_:tritonExpr) => some (.typed .boolean)
  | `(tritonExpr| tl.sin($a:tritonExpr)) | `(tritonExpr| tl.math.sin($a:tritonExpr))
  | `(tritonExpr| tl.cos($a:tritonExpr)) | `(tritonExpr| tl.exp($a:tritonExpr))
  | `(tritonExpr| tl.sigmoid($a:tritonExpr)) | `(tritonExpr| tl.rsqrt($a:tritonExpr))
  | `(tritonExpr| tl.math.rsqrt($a:tritonExpr))
  | `(tritonExpr| libdevice.exp($a:tritonExpr))
  | `(tritonExpr| tl.log($a:tritonExpr)) | `(tritonExpr| tl.sqrt($a:tritonExpr)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($a:tritonExpr, $_:num)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.sum($a:tritonExpr $[, $_:tritonReduceKwarg]*))
  | `(tritonExpr| tl.max($a:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| -$a:tritonExpr) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.trans($a:tritonExpr)) => valueType pointers values a
  | _ => none

private partial def containsStore (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "tl.store") || raw.getArgs.any containsStore

/-- The current ComputeExpr representation retains load precision at a complete
load assignment. Nested floating loads cannot yet carry that event through an
arbitrary algebraic expression. Refuse that form instead of dropping it. -/
private partial def rejectNestedLoads (raw : Syntax) (completeLoad : Bool := Bool.false) : MacroM Unit := do
  let isLoad := raw.getNumArgs > 0 && raw[0].isAtom && raw[0].getAtomVal == "tl.load("
  if isLoad && !completeLoad then
    Macro.throwErrorAt raw "nested load source precision is unsupported by the current frontend"
  for arg in raw.getArgs do rejectNestedLoads arg

private def checkAssignmentLoads (value : TSyntax `tritonExpr) : MacroM Unit := do
  -- Direct load-and-cast is a supported complete ComputeExpr as well.
  rejectNestedLoads value.raw Bool.true

private partial def rejectPointerRebinding (pointers : Types) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $id:ident = $value:tritonExpr) := raw then
    if let some dtype := lookup pointers id then
      unless pointerType pointers value == some dtype do
        Macro.throwErrorAt raw "pointer alias rebinding inside control flow requires an explicit supported merge"
  if let `(tritonStmt| $id:ident += $value:tritonExpr) := raw then
    if (lookup pointers id).isSome && (pointerType pointers value).isSome then
      Macro.throwErrorAt raw "a pointer update requires an integer offset"
  if let `(tritonStmt| $id:ident -= $value:tritonExpr) := raw then
    if (lookup pointers id).isSome && (pointerType pointers value).isSome then
      Macro.throwErrorAt raw "a pointer update requires an integer offset"
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) := raw then
    for id in #[a,b] ++ more do
      if (lookup pointers id).isSome then
        Macro.throwErrorAt raw "pointer tuple rebinding inside control flow is unsupported"
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

private partial def mentions (names : List Name) (raw : Syntax) : Bool :=
  (raw.isIdent && names.contains raw.getId.eraseMacroScopes) || raw.getArgs.any (mentions names)

private def checkInvariant (invariants : Values) (id : Ident) (value : Option ValueInfo) : MacroM Unit := do
  if let some (_, expected) := invariants.find? (fun p => p.1 == name id) then
    unless value == some expected do
      Macro.throwErrorAt id "loop-carried value requires a stable, known source dtype"

/-- Every store must resolve its destination even when its conversion is an
identity. Unknown control flow fails closed; branch results do not acquire
an unproved source dtype after the merge. -/
partial def lower (pointers : Types) (values : Values)
    (body : List (TSyntax `tritonStmt)) (invariants : Values := []) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | stmt :: rest =>
      let mut nextPointers := pointers
      let mut nextValues := values
      let mut stmt := stmt
      let loopBody (iterator : Ident) (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt) × Values) := do
        if (lookup pointers iterator).isSome || values.any (fun p => p.1 == name iterator) then
          Macro.throwErrorAt iterator "loop iterator shadows a tracked pointer or value"
        for st in body do rejectPointerRebinding pointers st.raw
        let stable := values.filter (fun p => match p.2 with | .typed _ => Bool.true | _ => Bool.false)
        let body ← lower pointers values body.toList (invariants ++ stable)
        return (body.toArray, stable)
      match stmt with
      | `(tritonStmt| $id:ident = $value:tritonExpr) =>
          checkAssignmentLoads value
          checkInvariant invariants id (valueType pointers values value)
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := pointerType pointers value then
            nextPointers := (name id, dtype) :: nextPointers
          if let some dtype := valueType pointers values value then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident += $value:tritonExpr) =>
          rejectNestedLoads value.raw
          let expression ← `(tritonExpr| $id:ident + $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident -= $value:tritonExpr) =>
          rejectNestedLoads value.raw
          let expression ← `(tritonExpr| $id:ident - $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident *= $value:tritonExpr) =>
          rejectNestedLoads value.raw
          let expression ← `(tritonExpr| $id:ident * $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident /= $value:tritonExpr) =>
          rejectNestedLoads value.raw
          let expression ← `(tritonExpr| $id:ident / $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
          rejectNestedLoads value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
          rejectNestedLoads value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          rejectNestedLoads condition.raw
          for st in yes ++ no do rejectPointerRebinding pointers st.raw
          let yes := (← lower pointers values yes.toList invariants).toArray
          let no := (← lower pointers values no.toList invariants).toArray
          stmt ← `(tritonStmt| if $condition { $yes* } else { $no* })
          nextValues := []
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          rejectNestedLoads condition.raw
          for st in yes do rejectPointerRebinding pointers st.raw
          let yes := (← lower pointers values yes.toList invariants).toArray
          stmt ← `(tritonStmt| if $condition { $yes* })
          nextValues := []
      | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($start, $stop, $step) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($start, $stop) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($stop) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in tl.static_range($n:num) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in tl.static_range($n:num) { $body* })
          nextValues := stable
      | _ =>
          if mentions (invariants.map Prod.fst) stmt.raw then
            Macro.throwErrorAt stmt "unsupported statement involving a loop-carried source dtype"
          rejectNestedLoads stmt.raw
          rejectPointerRebinding pointers stmt.raw
          if containsStore stmt.raw then
            Macro.throwErrorAt stmt "unsupported store control flow in call signature"
          -- An unsupported statement can assign a tracked value through a
          -- form not understood above. Forget value types conservatively.
          nextValues := []
      return stmt :: (← lower nextPointers nextValues rest invariants)

end VeriTile.Triton.DSL.TypedStores
