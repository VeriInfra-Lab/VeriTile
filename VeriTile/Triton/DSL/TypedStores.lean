import VeriTile.Triton.DSL.CallValue
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

private def name (id : Ident) : Name := id.getId.eraseMacroScopes

private def lookup (types : Types) (id : Ident) : Option Triton.SourceDType :=
  (types.find? (fun entry => entry.1 == name id)).map Prod.snd

/-- Block-pointer construction and advance preserve their base element type.
The expression compiler separately validates every shape/stride/order field. -/
private def blockPointerBase (e : TSyntax `tritonExpr) : Option (TSyntax `tritonExpr) := do
  let args := e.raw.getArgs
  let first ← args[0]?
  unless first.isAtom do failure
  if first.getAtomVal == "tl.make_block_ptr(" then
    if args[1]?.any Syntax.isIdent && args[2]?.any (fun a => a.isAtom && a.getAtomVal == "=") then
      unless args[1]!.getId.eraseMacroScopes == `base do failure
      return ⟨← args[3]?⟩
    return ⟨← args[1]?⟩
  if first.getAtomVal == "tl.advance(" then return ⟨← args[1]?⟩
  failure

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
  | _ => do
      let base ← blockPointerBase e
      pointerType types base

private partial def dtypeOverride (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "dtype") ||
  (raw.isIdent && raw.getId.eraseMacroScopes == `dtype) || raw.getArgs.any dtypeOverride

private def sourceKinds : List Triton.SourceDType :=
  [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean]

private partial def dtypeToken (raw : Syntax) : Option Name :=
  if raw.isAtom then
    (sourceKinds.find? (fun dtype => (SourceDType.dtypeName dtype).toString == raw.getAtomVal)).map SourceDType.dtypeName
  else if raw.isIdent then some raw.getId.eraseMacroScopes
  else if raw.getNumArgs == 1 then dtypeToken raw[0]
  else none

private def explicitCastType (pointers : Types) (e : TSyntax `tritonExpr) : Option Triton.SourceDType := do
  let direct : Option (TSyntax `tritonDType) := match e with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | _ => none
  if let some dt := direct then
    let target ← dtypeToken dt.raw
    return ← sourceKinds.find? (fun dtype => SourceDType.dtypeName dtype == target)
  unless (methodCast? e).isSome ||
      [``tritonMethodCastElementTy,``tritonMethodCastElementTyIdent,
       ``tritonIdentMethodCastElementTyIdent,``tritonIdentMethodCastElementTyIdentSpaced].contains e.raw.getKind ||
      e.raw.getArgs.any (fun arg => arg.isAtom && arg.getAtomVal == "to") do failure
  let target ← (methodCastTargetName? e).orElse fun _ => do
    let (_, dtype) ← methodCast? e
    dtypeToken dtype.raw
  match target with
  | .str (.str pointer "dtype") "element_ty"
  | .str (.str pointer "type") "element_ty" => lookup pointers (mkIdent pointer)
  | _ => sourceKinds.find? (fun dtype => SourceDType.dtypeName dtype == target)

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
    | .typed .fp64, .integerLiteral | .integerLiteral, .typed .fp64
    | .typed .fp64, .integerScalar | .integerScalar, .typed .fp64
    | .typed .fp64, .floatingLiteral | .floatingLiteral, .typed .fp64 => some .fp64
    | .typed .fp64, .typed other | .typed other, .typed .fp64 =>
        if [.fp64,.fp32,.fp16,.bf16,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains other
        then some .fp64 else none
    | .typed .fp32, .typed .fp32
    | .typed .fp32, .integerLiteral | .integerLiteral, .typed .fp32
    | .typed .fp32, .integerScalar | .integerScalar, .typed .fp32
    | .typed .fp32, .floatingLiteral | .floatingLiteral, .typed .fp32 => some .fp32
    | .typed .fp32, .typed integer | .typed integer, .typed .fp32 =>
        if [.fp16,.bf16,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains integer then some .fp32 else none
    | _, _ => none
  if let some dtype := (explicitCastType pointers e).orElse (fun _ => constructorType e) then some (.typed dtype) else
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
  | `(tritonExpr| tl.sum($a:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.max($a:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      match valueType pointers values a with
      | some (.typed .fp16) | some (.typed .bf16) | some (.typed .fp32) => some (.typed .fp32)
      | some (.typed .fp64) => some (.typed .fp64)
      | _ => none
  | `(tritonExpr| -$a:tritonExpr) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| $a:tritonExpr[:, None])
  | `(tritonExpr| $a:tritonExpr[None, :])
  | `(tritonExpr| $a:tritonExpr[None])
  | `(tritonExpr| $a:tritonExpr[:, None, None])
  | `(tritonExpr| $a:tritonExpr[None, :, None])
  | `(tritonExpr| $a:tritonExpr[None, None, :])
  | `(tritonExpr| $a:tritonExpr[:, None, :])
  | `(tritonExpr| $a:tritonExpr[None, :, :])
  | `(tritonExpr| $a:tritonExpr[:, :, None]) => valueType pointers values a
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      unless valueType pointers values a == some (.typed .fp32) &&
          valueType pointers values b == some (.typed .fp32) do failure
      let mut result := Triton.SourceDType.fp32
      for kw in kwargs do
        if let `(tritonReduceKwarg| out_dtype=$dtype:tritonDType) := kw then
          let target ← dtypeToken dtype.raw
          result ← sourceKinds.find? (fun kind => SourceDType.dtypeName kind == target)
      return .typed result
  | `(tritonExpr| tl.trans($a:tritonExpr)) => valueType pointers values a
  | _ => none

private partial def containsStore (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "tl.store") || raw.getArgs.any containsStore

/-- Integer memory reads have no floating load event. Identify the actual
source pointer type, rather than trusting a cast's destination type. -/
private def integerLoad (pointers : Types) (raw : Syntax) : Bool := Id.run do
  let e : TSyntax `tritonExpr := ⟨raw⟩
  let source : Option (TSyntax `tritonExpr × Array (TSyntax `tritonMemKwarg)) := match e with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) => some (p,kwargs)
    | _ => none
  let some (p,kwargs) := source | return Bool.false
  if kwargs.any (fun kw => dtypeOverride kw.raw) then return Bool.false
  return (pointerType pointers p).any (fun dtype =>
    [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains dtype)

/-- Complete floating load assignments retain their source precision. Nested
floating reads still fail closed; known integer reads can remain in algebraic
expressions. Check their address/mask children too, so a nested floating read
cannot be hidden by an integer outer load or a conversion to integer. -/
private partial def rejectNestedLoads (pointers : Types) (raw : Syntax)
    (completeLoad : Bool := Bool.false) : MacroM Unit := do
  let isLoad := raw.getNumArgs > 0 && raw[0].isAtom && raw[0].getAtomVal == "tl.load("
  if isLoad && !completeLoad && !integerLoad pointers raw then
    Macro.throwErrorAt raw "nested load source precision is unsupported by the current frontend"
  for arg in raw.getArgs do rejectNestedLoads pointers arg

private def checkAssignmentLoads (pointers : Types) (value : TSyntax `tritonExpr) : MacroM Unit := do
  -- The existing integer frontend preserves a direct tl.cast(load(...), dtype)
  -- just like load(...).to(dtype). Integer reads have no floating load event.
  -- Keep arbitrary nested floating loads rejected: their load precision must
  -- not disappear into an algebraic Op during source elaboration.
  if let `(tritonExpr| tl.cast($loaded:tritonExpr, $_:tritonDType)) := value then
    let pointer : Option (TSyntax `tritonExpr) := match loaded with
      | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
      | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
      | _ => none
    if let some p := pointer then
      if (pointerType pointers p).any (fun dt =>
          [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains dt) then
        rejectNestedLoads pointers loaded.raw Bool.true
        return
  -- Direct load-and-method-cast is a supported complete ComputeExpr as well.
  rejectNestedLoads pointers value.raw Bool.true

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

/-- Retain only names with the same known information on every path.
This tracks dtype, never equality of pointer addresses or computed values. -/
private def commonInfo [BEq α] (yes no : List (Name × α)) : List (Name × α) :=
  yes.filter (fun entry => no.any (fun other => other.1 == entry.1 && other.2 == entry.2))

/-- Lower the full block and retain its conservative exit information. A
supported branch joins only identical known dtypes from both exits; one-arm
branches include the unchanged entry environment as their other exit. -/
private partial def lowerWithState (pointers : Types) (values : Values)
    (body : List (TSyntax `tritonStmt)) (invariants : Values := []) :
    MacroM (List (TSyntax `tritonStmt) × Types × Values) := do
  match body with
  | [] => return ([],pointers,values)
  | stmt :: rest =>
      let mut nextPointers := pointers
      let mut nextValues := values
      let mut stmt := stmt
      let loopBody (iterator : Ident) (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt) × Values) := do
        if (lookup pointers iterator).isSome || values.any (fun p => p.1 == name iterator) then
          Macro.throwErrorAt iterator "loop iterator shadows a tracked pointer or value"
        for st in body do rejectPointerRebinding pointers st.raw
        -- This pass proves floating store precision, not integer SSA widths.
        -- Forget integer entry facts before lowering a possibly zero-trip loop:
        -- arithmetic can reassign those registers, and unknown integer widths
        -- must retain the destination conversion. Explicit source casts remain
        -- in the ordinary expression compiler. Floating carried facts still
        -- require a stable dtype on every assignment and every branch.
        let stable := values.filter (fun p => match p.2 with
          | .typed dtype => [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5].contains dtype
          | _ => Bool.false)
        let entry := values.filter (fun p => match p.2 with
          | .typed dtype => [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5].contains dtype
          | _ => Bool.true)
        let (body,_,_) ← lowerWithState pointers entry body.toList (invariants ++ stable)
        return (body.toArray, stable)
      match stmt with
      | `(tritonStmt| $a:ident, $b:ident $[, $ids:ident]* =
          $x:tritonExpr, $y:tritonExpr $[, $rhs:tritonExpr]*) =>
          let targets := #[a,b] ++ ids
          let expressions := #[x,y] ++ rhs
          unless targets.size == expressions.size do
            Macro.throwErrorAt stmt "tuple assignment needs one value per target"
          if targets.any (fun id => invariants.any (fun entry => entry.1 == name id)) then
            Macro.throwErrorAt stmt "unsupported statement involving a loop-carried source dtype"
          -- Python evaluates all RHSs in the entry environment. Preserve
          -- unrelated ABI facts, and replace only the explicitly bound names.
          for (id,value) in targets.zip expressions do
            if (lookup pointers id).isSome || (pointerType pointers value).isSome then
              Macro.throwErrorAt stmt "pointer tuple assignment needs an explicit supported binding"
            checkAssignmentLoads pointers value
            let dtype := valueType pointers values value
            checkInvariant invariants id dtype
            nextPointers := nextPointers.filter (fun entry => entry.1 != name id)
            nextValues := nextValues.filter (fun entry => entry.1 != name id)
            if let some dtype := dtype then
              nextValues := (name id, dtype) :: nextValues
      -- The grammar has a dedicated positional-axis assignment production.
      -- Its reduction dtype must enter the same dataflow as a generic RHS;
      -- otherwise a later store spuriously adds another destination cast.
      | `(tritonStmt| $id:ident = tl.max($value:tritonExpr, $axisIndex:num))
      | `(tritonStmt| $id:ident := tl.max($value:tritonExpr, $axisIndex:num)) =>
          let reduction ← `(tritonExpr| tl.max($value:tritonExpr, axis = $axisIndex:num))
          checkAssignmentLoads pointers reduction
          let dtype := valueType pointers values reduction
          checkInvariant invariants id dtype
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := dtype then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident = $value:tritonExpr) =>
          checkAssignmentLoads pointers value
          checkInvariant invariants id (valueType pointers values value)
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := pointerType pointers value then
            nextPointers := (name id, dtype) :: nextPointers
          if let some dtype := valueType pointers values value then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident += $value:tritonExpr) =>
          rejectNestedLoads pointers value.raw
          let expression ← `(tritonExpr| $id:ident + $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident -= $value:tritonExpr) =>
          rejectNestedLoads pointers value.raw
          let expression ← `(tritonExpr| $id:ident - $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident *= $value:tritonExpr) =>
          rejectNestedLoads pointers value.raw
          let expression ← `(tritonExpr| $id:ident * $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident /= $value:tritonExpr) =>
          rejectNestedLoads pointers value.raw
          let expression ← `(tritonExpr| $id:ident / $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
          rejectNestedLoads pointers value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
          rejectNestedLoads pointers value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          rejectNestedLoads pointers condition.raw
          for st in yes ++ no do rejectPointerRebinding pointers st.raw
          let (yes,yesPointers,yesValues) ← lowerWithState pointers values yes.toList invariants
          let (no,noPointers,noValues) ← lowerWithState pointers values no.toList invariants
          let yes := yes.toArray
          let no := no.toArray
          stmt ← `(tritonStmt| if $condition { $yes* } else { $no* })
          nextPointers := commonInfo yesPointers noPointers
          nextValues := commonInfo yesValues noValues
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          rejectNestedLoads pointers condition.raw
          for st in yes do rejectPointerRebinding pointers st.raw
          let (yes,yesPointers,yesValues) ← lowerWithState pointers values yes.toList invariants
          let yes := yes.toArray
          stmt ← `(tritonStmt| if $condition { $yes* })
          nextPointers := commonInfo yesPointers pointers
          nextValues := commonInfo yesValues values
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
      | `(tritonStmt| for $i:ident in tl.static_range($($n:term)) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in tl.static_range($($n)) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in tl.static_range($n:num) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in tl.static_range($n:num) { $body* })
          nextValues := stable
      | _ =>
          if mentions (invariants.map Prod.fst) stmt.raw then
            Macro.throwErrorAt stmt "unsupported statement involving a loop-carried source dtype"
          rejectNestedLoads pointers stmt.raw
          rejectPointerRebinding pointers stmt.raw
          if containsStore stmt.raw then
            Macro.throwErrorAt stmt "unsupported store control flow in call signature"
          -- An unsupported statement can assign a tracked value through a
          -- form not understood above. Forget value types conservatively.
          nextValues := []
      let (rest,finalPointers,finalValues) ← lowerWithState nextPointers nextValues rest invariants
      return (stmt :: rest,finalPointers,finalValues)

/-- Every store resolves its destination dtype. Supported control flow keeps
only dtype facts proved by both exits; unsupported forms forget value facts. -/
def lower (pointers : Types) (values : Values)
    (body : List (TSyntax `tritonStmt)) (invariants : Values := []) : MacroM (List (TSyntax `tritonStmt)) := do
  return (← lowerWithState pointers values body invariants).1

end VeriTile.Triton.DSL.TypedStores
