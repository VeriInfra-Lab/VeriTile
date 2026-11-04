import VeriTile.Triton.DSL.SignedCallRanges
import VeriTile.Triton.DSL.SignedIntegerAddressAliases

/-! Opt-in implicit integer promotion for augmented assignments. Only an
already signed64 integer lhs and a known integral rhs select this path.
Floating, pointer and unknown operands keep the ordinary elaboration. -/
namespace VeriTile.Triton.DSL.SignedAugmentedAssignments
open Lean Elab Term Meta
private abbrev Types := List (Name × Nat)

private partial def kind (types : Types) (value : TSyntax `tritonExpr) : Nat :=
  if value.raw.getKind==choiceKind then kind types ⟨value.raw[0]!⟩
  else match value with
  | `(tritonExpr| $name:ident[:, None])
  | `(tritonExpr| $name:ident[None, :]) => (types.lookup name.getId.eraseMacroScopes).getD 0
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
    let ka:=kind types a; let kb:=kind types b
    if ka>0 && kb>0 then max ka kb else 0
  | _ => SignedIntegerAddressAliases.kind types value

private def meet (left right : Types) : Types :=
  left.filter (fun (name,k)=>right.lookup name==some k)

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

/-- An unhandled write must not retain signed evidence from an earlier value.
Intrinsic keyword nodes cannot establish new type evidence. -/
private partial def writes (raw : Syntax) : List Name :=
  let args:=raw.getArgs.toList
  let leaf:=match raw.getKind with | .str _ name=>name | _=>""
  let own:=if leaf.startsWith "tritonStmt" || ["tupleValues","helperTupleCall"].contains leaf then
      match args.findIdx? (fun token=>token.isAtom &&
          ["=",":=","+=","-=","*=","/="].contains token.getAtomVal.trimAscii.toString) with
      | some i=>(args.take i).flatMap ids
      | none=>[]
    else []
  own++raw.getArgs.toList.flatMap writes

private partial def lower (types : Types) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt) × Types) := do
  let mut types:=types
  let mut result:=#[]
  for original in body do
    let statement : TSyntax `tritonStmt:=
      if original.raw.getKind==choiceKind then ⟨original.raw[0]!⟩ else original
    let statement ← match statement with
      | `(tritonStmt| $name:ident = $value:tritonExpr)=>do
        let k:=kind types value
        types:=SignedIntegerAddressAliases.bindName types name.getId.eraseMacroScopes k
        -- Retain the inferred signed type when downstream address pinning runs.
        if k==2 then `(tritonStmt| $name:ident = tl.cast($value:tritonExpr, tl.int64))
        else pure statement
      | `(tritonStmt| $name:ident += $value:tritonExpr)=>do
        let lhs:=(types.lookup name.getId.eraseMacroScopes).getD 0
        let rhs:=kind types value
        types:=SignedIntegerAddressAliases.bindName types name.getId.eraseMacroScopes
          (if lhs>0 && rhs>0 then max lhs rhs else 0)
        if lhs==2 && rhs==1 then
          `(tritonStmt| $name:ident += tl.cast($value:tritonExpr, tl.int64))
        else pure statement
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>do
        let (yes,yt) ← lower types yes
        let (no,nt) ← lower types no
        types:=meet yt nt
        `(tritonStmt| if $condition { $yes* } else { $no* })
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* })=>do
        let (yes,yt) ← lower types yes
        types:=meet types yt
        `(tritonStmt| if $condition { $yes* })
      | `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr, $step:tritonExpr) { $nested:tritonStmt* })=>do
        let k:=if kind types hi==2 then 2 else 1
        let (nested,nt) ← lower (SignedIntegerAddressAliases.bindName types counter.getId.eraseMacroScopes k) nested
        types:=meet types nt
        `(tritonStmt| for $counter:ident in range(0, $hi, $step) { $nested* })
      | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $step:tritonExpr) { $nested:tritonStmt* })=>do
        let k:=if kind types lo==2 || kind types hi==2 then 2 else 1
        let (nested,nt) ← lower (SignedIntegerAddressAliases.bindName types counter.getId.eraseMacroScopes k) nested
        types:=meet types nt
        `(tritonStmt| for $counter:ident in range($lo, $hi, $step) { $nested* })
      | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr) { $nested:tritonStmt* })=>do
        let k:=if kind types lo==2 || kind types hi==2 then 2 else 1
        let (nested,nt) ← lower (SignedIntegerAddressAliases.bindName types counter.getId.eraseMacroScopes k) nested
        types:=meet types nt
        `(tritonStmt| for $counter:ident in range($lo, $hi) { $nested* })
      | `(tritonStmt| for $counter:ident in range($hi:tritonExpr) { $nested:tritonStmt* })=>do
        let k:=if kind types hi==2 then 2 else 1
        let (nested,nt) ← lower (SignedIntegerAddressAliases.bindName types counter.getId.eraseMacroScopes k) nested
        types:=meet types nt
        `(tritonStmt| for $counter:ident in range($hi) { $nested* })
      | _=>do
        let changed:=writes statement.raw
        types:=types.filter (fun entry=>!changed.contains entry.1)
        pure statement
    result:=result.push statement
  return (result,types)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let stx ← PythonFullConstructors.prepareSource stx
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let types ← SignedIntegerAddressAliases.initialKinds profile
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw message
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwErrorAt raw message
  let `(triton { $body:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let (body,_) ← liftMacroM (lower types body)
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  SignedCallRanges.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.SignedAugmentedAssignments
