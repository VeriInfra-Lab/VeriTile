import VeriTile.Triton.DSL.Expansion.Common

/-! Checked declaration and lowering of the pinned finite-classification
helper. Source metadata retains its complete return expression; only exact
parameter/alias use is admitted. This pass changes calls, never control flow. -/
open Lean
namespace VeriTile.Triton.DSL.FiniteHelperSource

private structure Context where
  aliases : List (String × FinitePrecision) := []
  helpers : List String := []

private def key (i : TSyntax `ident) : String := i.getId.eraseMacroScopes.toString

private def names (ctx : Context) : List String := ctx.aliases.map Prod.fst ++ ctx.helpers

private def rejectName (ctx : Context) (i : TSyntax `ident) : MacroM Unit := do
  let name := key i
  if (names ctx).any (fun n => name == n || name.startsWith (n ++ ".")) then
    Macro.throwError "finite helper and extern names cannot be rebound, aliased, or used outside direct calls"

private def fresh (ctx : Context) (i : TSyntax `ident) : MacroM Unit := do
  let .str .anonymous _ := i.getId.eraseMacroScopes
    | Macro.throwError "finite source bindings require simple source identifiers"
  if (names ctx).contains (key i) || key i == "tl" || key i == "triton" then
    Macro.throwError "finite source binding must be fresh and unambiguous"

private def freshBinding (ctx : Context) (externalNames : List String)
    (i : TSyntax `ident) : MacroM Unit := do
  fresh ctx i
  if externalNames.contains (key i) then
    Macro.throwError "finite source binding captures an outer Lean parameter or local binding"

private def aliasPrecision (target : TSyntax `ident) : MacroM FinitePrecision := do
  match target.getId.eraseMacroScopes with
  | `triton.language.extra.cuda.libdevice.finitef => return .fp32
  | `triton.language.extra.cuda.libdevice.isfinited => return .fp64
  | _ => Macro.throwError "finite source alias requires the explicit CUDA libdevice finitef/isfinited import"

private def requireParameter (parameter : TSyntax `ident) (e : TSyntax `tritonExpr) : MacroM Unit := do
  if e.raw.isIdent then
    unless key ⟨e.raw⟩ == key parameter do
      Macro.throwError "finite helper must use its declared parameter unchanged"
    return
  match e with
  | `(tritonExpr| $i:ident) =>
      unless key i == key parameter do
        Macro.throwError "finite helper must use its declared parameter unchanged"
  | _ => Macro.throwError "finite helper must use its declared parameter unchanged"

private def dtypeSubject? (condition : TSyntax `tritonExpr) : Option (TSyntax `ident) := do
  let args := condition.raw.getArgs
  let first ← args[0]?
  if !first.isIdent then none
  else if condition.raw.getKind == ``tritonSourceIsFP64Dotted then
    match first.getId.eraseMacroScopes with
    | .str (.str subject "dtype") "is_fp64" => some (mkIdent subject)
    | _ => none
  else none

private def requireAlias (ctx : Context) (name : TSyntax `ident)
    (p : FinitePrecision) : MacroM Unit := do
  unless ctx.aliases.any (fun entry => entry.1 == key name && entry.2 == p) do
    Macro.throwError "finite helper call lacks the matching declared source extern alias"

private def checkHelper (ctx : Context) (parameter : TSyntax `ident)
    (body : TSyntax `tritonExpr) : MacroM Unit := do
  let `(tritonExpr| ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr)) := body
    | Macro.throwError "finite helper requires its exact fp64 dtype-dispatch return expression"
  let some subject := dtypeSubject? condition
    | Macro.throwError "finite helper requires the parameter dtype.is_fp64 predicate"
  unless key subject == key parameter do
    Macro.throwError "finite helper dtype predicate must inspect its declared parameter"
  let `(tritonExpr| $wide:ident($wideArg:tritonExpr)) := yes
    | Macro.throwError "finite helper fp64 branch must call the declared isfinited alias"
  requireAlias ctx wide .fp64
  requireParameter parameter wideArg
  let `(tritonExpr| $narrow:ident($narrowArg:tritonExpr)) := no
    | Macro.throwError "finite helper narrow branch must call the declared finitef alias"
  requireAlias ctx narrow .fp32
  let some (castArg, castDType) := methodCast? narrowArg
    | Macro.throwError "finite helper narrow branch requires its explicit float32 cast"
  let `(tritonDType| tl.float32) := castDType
    | Macro.throwError "finite helper narrow branch requires its explicit float32 cast"
  requireParameter parameter castArg

private partial def containsFeature (raw : Syntax) : Bool :=
  raw.getKind == ``tritonSourceFiniteAlias || raw.getKind == ``tritonSourceFiniteHelper ||
  raw.getKind == ``tritonFiniteHelperResolved || raw.getArgs.any containsFeature

/-- A dotted `value.to(pointer.type.element_ty)` can parse as a generic
helper call because its target is not a literal tritonDType. Recover only
this exact method spelling before RandomProfile visits generic calls. The
normal pointer-provenance pass still validates and retains the conversion. -/
private def pointerElementCast? (ctx : Context) (e : TSyntax `tritonExpr) :
    MacroM (Option (TSyntax `tritonExpr)) := do
  -- Direct load casts have their own parser node, which otherwise hides both
  -- the load and its pointer-element conversion from storage annotation.
  let loadCast? ← match e with
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($targetName:ident)) =>
        pure (some (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)), targetName))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($targetName:ident)) =>
        pure (some (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)), targetName))
    | _ => pure none
  if let some (loaded, targetName) := loadCast? then
    let .str (.str pointerName "type") "element_ty" := targetName.getId.eraseMacroScopes
      | return none
    let .str .anonymous _ := pointerName | return none
    rejectName ctx targetName
    let target ← `(term| $targetName:ident)
    return some (← `(tritonExpr| ($loaded:tritonExpr).to($target:term)))
  match e with
  | `(tritonExpr| $methodName:ident($argument:tritonExpr)) =>
      let .str valueName "to" := methodName.getId.eraseMacroScopes | return none
      let .str .anonymous _ := valueName | return none
      let `(tritonExpr| $targetName:ident) := argument | return none
      let .str (.str pointerName "type") "element_ty" := targetName.getId.eraseMacroScopes
        | return none
      let .str .anonymous _ := pointerName | return none
      let valueIdent : Ident := mkIdentFrom methodName valueName
      rejectName ctx valueIdent
      rejectName ctx targetName
      let value ← `(tritonExpr| $valueIdent:ident)
      let target ← `(term| $targetName:ident)
      return some (← `(tritonExpr| ($value:tritonExpr).to($target:term)))
  | _ => return none

mutual

private partial def lowerArgument (ctx : Context) (argument : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr) := do
  return ⟨← lowerRaw ctx argument.raw⟩

private partial def directCall? (ctx : Context) (e : TSyntax `tritonExpr) :
    MacroM (Option (TSyntax `tritonExpr)) := do
  match e with
  | `(tritonExpr| $helper:ident($argument:tritonExpr)) =>
      if ctx.helpers.contains (key helper) then
        let argument ← lowerArgument ctx argument
        return some (← `(tritonExpr| tl.__finite_helper_resolved($argument:tritonExpr)))
      match ctx.aliases.find? (fun entry => entry.1 == key helper) with
      | some (_, .fp32) =>
          let argument ← lowerArgument ctx argument
          return some (← `(tritonExpr| tl.extra.cuda.libdevice.finitef($argument:tritonExpr)))
      | some (_, .fp64) =>
          let argument ← lowerArgument ctx argument
          return some (← `(tritonExpr| tl.extra.cuda.libdevice.isfinited($argument:tritonExpr)))
      | none => return none
  | _ => return none

private partial def lowerRaw (ctx : Context) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == ``tritonFiniteHelperResolved then
    Macro.throwError "finite helper resolved calls are internal syntax"
  if raw.getKind == ``tritonSourceFiniteAlias || raw.getKind == ``tritonSourceFiniteHelper then
    Macro.throwError "finite source declarations must precede the writer body at top level"
  if let some converted ← pointerElementCast? ctx ⟨raw⟩ then
    -- The normalized method form is no longer recognized above. Still visit
    -- its original load pointer/kwargs so protected/internal uses cannot hide.
    return ← lowerRaw ctx converted.raw
  let st : TSyntax `tritonStmt := ⟨raw⟩
  match st with
  | `(tritonStmt| $destination:ident = $value:tritonExpr) =>
      if let some lowered ← directCall? ctx value then
        rejectName ctx destination
        return (← `(tritonStmt| $destination:ident = $lowered:tritonExpr)).raw
  | `(tritonStmt| $destination:ident := $value:tritonExpr) =>
      if let some lowered ← directCall? ctx value then
        rejectName ctx destination
        return (← `(tritonStmt| $destination:ident := $lowered:tritonExpr)).raw
  | _ => pure ()
  if raw.isIdent then rejectName ctx ⟨raw⟩
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (lowerRaw ctx))
  | other => return other

end

/-- Source declarations are a prefix. The remainder is traversed structurally,
preserving every statement and both arms of every control construct. Unknown
dtype evidence is rejected later at the actual call's expansion point. -/
def prepare (stmts : List (TSyntax `tritonStmt)) (externalNames : List String := []) :
    MacroM (List (TSyntax `tritonStmt)) := do
  if !stmts.any (fun st => containsFeature st.raw) then return stmts
  let mut ctx : Context := {}
  let mut body : List (TSyntax `tritonStmt) := []
  let mut started := Bool.false
  for st in stmts do
    match st with
    | `(tritonStmt| tl.source_alias $aliasName:ident = $target:ident) =>
        if started then
          Macro.throwError "finite source declarations must precede the writer body at top level"
        freshBinding ctx externalNames aliasName
        let precision ← aliasPrecision target
        ctx := { ctx with aliases := ctx.aliases ++ [(key aliasName, precision)] }
    | `(tritonStmt| tl.source_helper $helper:ident($parameter:ident) { return $expression:tritonExpr }) =>
        if started then
          Macro.throwError "finite source declarations must precede the writer body at top level"
        freshBinding ctx externalNames helper
        fresh ctx parameter
        unless key parameter != key helper do
          Macro.throwError "finite helper parameter cannot shadow its function name"
        checkHelper ctx parameter expression
        ctx := { ctx with helpers := ctx.helpers ++ [key helper] }
    | _ =>
        started := Bool.true
        body := body ++ [st]
  return ← body.mapM fun st => return ⟨← lowerRaw ctx st.raw⟩

end VeriTile.Triton.DSL.FiniteHelperSource
