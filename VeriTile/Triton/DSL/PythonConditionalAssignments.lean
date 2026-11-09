import VeriTile.Triton.DSL.BooleanCallControl
import VeriTile.Triton.DSL.PythonConditionGrouping
import VeriTile.Triton.DSL.PythonFullConstructors

/-! Copied Python assignment conditionals. Group only an assignment RHS for
parsing. Literal constexpr choices select their original expression; runtime
choices remain ordinary typed expressions. Boolean branch specialization is
provided by the existing immutable-constexpr adapter. -/
namespace VeriTile.Triton.DSL.PythonConditionalAssignments
open Lean Elab Term Meta

private def groupLine (line : String) : Except String String := do
  for operator in [" = "," += "," -= "," *= "," /= "] do
    let [left,right] := line.splitOn operator | continue
    let target:=left.trimAscii.toString
    unless !target.isEmpty && target.toList.all (fun c=>c.isAlphanum || c=='_' || c=='«' || c=='»') do continue
    let parts:=PythonConditionGrouping.splitWord right "if"
    if parts.length==1 then return line
    unless parts.length==2 && (PythonConditionGrouping.splitWord right "else").length==2 do
      throw "assignment conditional requires one complete if/else expression"
    return left++operator++"("++right++")"
  return line

private partial def boolean? (value : TSyntax `tritonExpr) : Option Bool :=
  match value with
  | `(tritonExpr| true)=>some Bool.true
  | `(tritonExpr| false)=>some Bool.false
  | `(tritonExpr| ($inner:tritonExpr))=>boolean? inner
  | `(tritonExpr| not $inner:tritonExpr)=>(!·) <$> boolean? inner
  | _=>none

private partial def hasLoad (raw : Syntax) : Bool :=
  let loaded := match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*))=>Bool.true
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*))=>Bool.true
    | _=>Bool.false
  loaded || raw.getArgs.any hasLoad

private partial def select (raw : Syntax) : TermElabM Syntax := do
  if raw.getKind==choiceKind then return ← select raw[0]!
  if let `(tritonExpr| tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType)) := raw then
    let floating := match dtype with
      | `(tritonDType| tl.float32) | `(tritonDType| tl.float16)
      | `(tritonDType| tl.bfloat16) | `(tritonDType| tl.float64)=>Bool.true
      | _=>Bool.false
    if floating then return ← select (← `(tritonExpr| (tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dtype:tritonDType))).raw
  if let `(tritonExpr| tl.load($pointer:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType)) := raw then
    let floating := match dtype with
      | `(tritonDType| tl.float32) | `(tritonDType| tl.float16)
      | `(tritonDType| tl.bfloat16) | `(tritonDType| tl.float64)=>Bool.true
      | _=>Bool.false
    if floating then return ← select (← `(tritonExpr| (tl.load($pointer:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dtype:tritonDType))).raw
  let raw:=raw.setArgs (← raw.getArgs.mapM select)
  if let `(tritonStmt| $id:ident += $value:tritonExpr) := raw then
    if hasLoad value.raw then return (← `(tritonStmt| $id:ident = $id:ident + $value:tritonExpr)).raw
  if let `(tritonStmt| $id:ident -= $value:tritonExpr) := raw then
    if hasLoad value.raw then return (← `(tritonStmt| $id:ident = $id:ident - $value:tritonExpr)).raw
  if let `(tritonStmt| $id:ident *= $value:tritonExpr) := raw then
    if hasLoad value.raw then return (← `(tritonStmt| $id:ident = $id:ident * $value:tritonExpr)).raw
  if let `(tritonStmt| $id:ident /= $value:tritonExpr) := raw then
    if hasLoad value.raw then return (← `(tritonStmt| $id:ident = $id:ident / $value:tritonExpr)).raw
  if let `(tritonStmt| for $index:ident in range(0, $stop:tritonExpr) { $body:tritonStmt* }) := raw then
    return (← `(tritonStmt| for $index in range($((0 : Nat)), $stop, $((1 : Nat))) { $body* })).raw
  if let `(tritonStmt| for $index:ident in range(0, $stop:tritonExpr, $stride:tritonExpr) { $body:tritonStmt* }) := raw then
    return (← `(tritonStmt| for $index in range($((0 : Nat)), $stop, $stride) { $body* })).raw
  if let `(tritonStmt| for $index:ident in range($stop:tritonExpr) { $body:tritonStmt* }) := raw then
    return (← `(tritonStmt| for $index in range($((0 : Nat)), $stop, $((1 : Nat))) { $body* })).raw
  if let `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) := raw then
    return (← `(tritonStmt| for $index in range($start, $stop, 1) { $body* })).raw
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr))=>
    match boolean? condition with
    | some b=>return if b then yes.raw else no.raw
    | none=>return raw
  | _=>return raw

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw message
  let lines ← (source.splitOn "\n").mapM fun line=>match groupLine line with
    | .ok value=>pure value | .error message=>throwErrorAt raw message
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames (String.intercalate "\n" lines)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw message
  let parsed ← select parsed
  let `(triton { $body:tritonStmt* }) := parsed | throwError "expected complete conditional source"
  let body ← liftMacroM (RawBodyNestedLoads.lower body.toList)
  let source:=String.intercalate "\n" (body.map (PythonFullConstructors.render ·.raw))
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

private partial def booleanArgument (names : Expr) : TermElabM Bool := do
  let names ← whnf names
  if names.isAppOfArity ``List.nil 1 then return Bool.false
  unless names.isAppOfArity ``List.cons 3 do throwError "constexpr names must be a static list"
  let .lit (.strVal name) ← whnf names.getAppArgs[1]! | throwError "constexpr name must be a string"
  let value ← elabTerm (mkIdent name.toName) none
  if ← isDefEq (← inferType value) (mkConst ``Bool) then return Bool.true
  booleanArgument names.getAppArgs[2]!

private partial def pointer (names : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $id:ident)=>names.contains id.getId.eraseMacroScopes
  | `(tritonExpr| ($inner:tritonExpr))=>pointer names inner
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $base:tritonExpr - $_:tritonExpr)=>pointer names base
  | _=>Bool.false

private partial def assignments (raw : Syntax) : List Name :=
  match raw with
  | `(tritonStmt| $id:ident = $_:tritonExpr)=>[id.getId.eraseMacroScopes]
  | _=>raw.getArgs.toList.flatMap assignments

/-- Capture a pointer prefix before its literal decrement. The original
single assignment and this two-instruction lowering have the same final
registers, memory and failures; the subtrahend cannot read the assigned name. -/
private partial def splitDecrements (names : List Name) (body : Array (TSyntax `tritonStmt)) :
    TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut names:=names
  let mut out:=#[]
  for original in body do
    let mut statement:=original
    match original with
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* })=>
      let nested ← splitDecrements (names.filter (· != index.getId.eraseMacroScopes)) nested
      statement ← `(tritonStmt| for $index in range($start, $stop, $stride) { $nested* })
      names:=names.filter (fun name=> !(assignments original.raw).contains name)
    | _=>pure ()
    if let `(tritonStmt| $id:ident = $value:tritonExpr) := statement then
      if let `(tritonExpr| $base:tritonExpr - $n:num) := value then
        if pointer names base then
          out:=out.push (← `(tritonStmt| $id:ident = $base:tritonExpr))
          out:=out.push (← `(tritonStmt| $id:ident -= $n:num))
          names:=id.getId.eraseMacroScopes::names.filter (· != id.getId.eraseMacroScopes)
          continue
      let isPointer:=pointer names value
      names:=names.filter (· != id.getId.eraseMacroScopes)
      if isPointer then names:=id.getId.eraseMacroScopes::names
    out:=out.push statement
  return out

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "caller metadata must be a static list"
  return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)

private def decrementAliases (stx : Syntax) (profile : Expr) : TermElabM Syntax := do
  let mut names:=[]
  for field in [``VeriTile.Triton.CallSignature.pointers,``VeriTile.Triton.CallSignature.floatingPointers] do
    for entry in ← listValues (← mkAppM field #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry])
        | throwError "pointer name must be a string"
      names:=Name.mkSimple name::names
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw message
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw message
  let `(triton { $body:tritonStmt* }) := parsed | throwError "expected pointer source"
  let body ← splitDecrements names body
  let source:=String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab CallSignature.withCall]
def elaborateCall : TermElab := fun stx expected=> do
  let stx ← prepare stx
  let profile ← elabTermEnsuringType stx[1] (mkConst ``VeriTile.Triton.CallSignature)
  let names ← mkAppM ``VeriTile.Triton.CallSignature.constexpr #[profile]
  if ← booleanArgument names then BooleanCallControl.elaborateInvocation stx expected
  else PythonInlineSuites.elaborateInvocation (← decrementAliases stx profile) expected

@[term_elab CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected=> do
  PythonInlineSuites.elaborateSignature (← prepare stx) expected
end VeriTile.Triton.DSL.PythonConditionalAssignments
