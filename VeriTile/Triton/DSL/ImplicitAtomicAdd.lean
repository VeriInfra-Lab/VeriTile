import VeriTile.Triton.DSL.SignedCallRanges

/-! Atomic-add operands undergo the destination pointer's implicit conversion,
just like Triton stores. Keep the pasted body and its control flow unchanged;
insert only the typed conversion before ordinary source elaboration. -/
namespace VeriTile.Triton.DSL.ImplicitAtomicAdd
open Lean Elab Term Meta

private partial def pointer (aliases : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $id:ident) => aliases.contains id.getId.eraseMacroScopes
  | `(tritonExpr| ($e:tritonExpr)) => pointer aliases e
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr) => pointer aliases base
  | _ => Bool.false

private partial def rejectRebind (aliases : List Name) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $id:ident = $value:tritonExpr) := raw then
    if aliases.contains id.getId.eraseMacroScopes && !pointer aliases value then
      Macro.throwErrorAt id "atomic pointer alias is rebound under runtime control flow"
  for child in raw.getArgs do rejectRebind aliases child

private partial def lower (dtype : TSyntax `tritonDType) (aliases : List Name)
    (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | statement::tail =>
    let mut statement:=statement
    let mut next:=aliases
    match statement with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      next:=aliases.filter (·!=id.getId.eraseMacroScopes)
      if pointer aliases value then next:=id.getId.eraseMacroScopes::next
    | `(tritonStmt| tl.atomic_add($destination:tritonExpr, $value:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if pointer aliases destination then
        statement ← `(tritonStmt| tl.atomic_add($destination:tritonExpr,
          tl.implicit_store_cast($value:tritonExpr, $dtype:tritonDType) $[, $kwargs:tritonMemKwarg]*))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      for st in yes++no do rejectRebind aliases st.raw
      let yes:=(← lower dtype aliases yes.toList).toArray
      let no:=(← lower dtype aliases no.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* })
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      for st in yes do rejectRebind aliases st.raw
      let yes:=(← lower dtype aliases yes.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* })
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $inner:tritonStmt* }) =>
      for st in inner do rejectRebind aliases st.raw
      let inner:=(← lower dtype aliases inner.toList).toArray
      statement ← `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $inner* })
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr) { $inner:tritonStmt* }) =>
      for st in inner do rejectRebind aliases st.raw
      let inner:=(← lower dtype aliases inner.toList).toArray
      statement ← `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr) { $inner* })
    | `(tritonStmt| for $index:ident in range($stop:tritonExpr) { $inner:tritonStmt* }) =>
      for st in inner do rejectRebind aliases st.raw
      let inner:=(← lower dtype aliases inner.toList).toArray
      statement ← `(tritonStmt| for $index:ident in range($stop:tritonExpr) { $inner* })
    | _ => pure ()
    return statement::(← lower dtype next tail)

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "atomic pointer types require a static call signature"

private partial def checkConverted (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| tl.atomic_add($_:tritonExpr, $value:tritonExpr $[, $_:tritonMemKwarg]*)) := raw then
    match value with
    | `(tritonExpr| tl.implicit_store_cast($_:tritonExpr, $_:tritonDType)) => pure ()
    | _ => Macro.throwErrorAt value "cannot determine tl.atomic_add destination dtype from the call signature"
  for child in raw.getArgs do checkConverted child

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body => pure body | .error message => throwErrorAt raw "{message}"
  let `(triton { $body:tritonStmt* }) := parsed | throwError "complete caller body required"
  let mut body:=body.toList
  for entry in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry])
      | throwError "pointer names must be literal"
    let dtype ← whnf (← mkAppM ``Prod.snd #[entry])
    let kinds : List (Name × SourceDType) :=
      [(``SourceDType.fp16,.fp16),(``SourceDType.bf16,.bf16),(``SourceDType.fp32,.fp32),
       (``SourceDType.fp64,.fp64),(``SourceDType.i8,.i8),(``SourceDType.i16,.i16),
       (``SourceDType.i32,.i32),(``SourceDType.i64,.i64)]
    if let some (_,kind) := kinds.find? (fun p=>dtype.isConstOf p.1) then
      let some dtype ← liftMacroM <| dtypeSyntaxOfName? (SourceDType.dtypeName kind)
        | throwError "unsupported atomic destination dtype"
      body ← liftMacroM <| lower dtype [name.toName] body
  for statement in body do liftMacroM (checkConverted statement.raw)
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (body.map (PythonFullConstructors.render ·.raw))++"\n}")
  SignedCallRanges.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.ImplicitAtomicAdd
