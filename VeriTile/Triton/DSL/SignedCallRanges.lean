import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.NamedFloatDTypeCasts

/-! A range with a known signed64 bound has a signed64 start/counter.
Keep the pasted range spelling; supply that promotion before
invocation substitution hides the source register behind an arithmetic cast.
Unknown or floating evidence never selects the signed path. -/
namespace VeriTile.Triton.DSL.SignedCallRanges
open Lean Elab Term Meta
private abbrev Types := List (Name × Nat)

private partial def kind (types : Types) (e : TSyntax `tritonExpr) : Nat :=
  if e.raw.getKind==choiceKind then kind types ⟨e.raw[0]!⟩
  else if let some (_,dt):=methodCast? e then
    match dt with | `(tritonDType| tl.int64) => 2 | _ => 0
  else match e with
  | `(tritonExpr| $id:ident) => (types.lookup id.getId.eraseMacroScopes).getD 0
  | `(tritonExpr| $n:num) => if n.getNat<2^63 then 1 else 0
  | `(tritonExpr| ($e:tritonExpr)) | `(tritonExpr| -$e:tritonExpr) => kind types e
  | `(tritonExpr| tl.cast($_:tritonExpr, tl.int64)) => 2
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr)
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
    let a:=kind types a;let b:=kind types b
    if a>0 && b>0 then max a b else 0
  | _ => 0

private partial def writes (raw : Syntax) : List Name :=
  let args:=raw.getArgs
  let direct := if args.size≥2 && args[0]!.isIdent && args[1]!.isAtom &&
      ["=",":=","+=","-=","*=","/="].contains args[1]!.getAtomVal.trimAscii.toString
    then [args[0]!.getId.eraseMacroScopes] else []
  let binder := if args.size≥2 && args[0]!.isAtom && args[0]!.getAtomVal.trimAscii.toString=="for" && args[1]!.isIdent
    then [args[1]!.getId.eraseMacroScopes] else []
  let tuple := match raw with
    | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
      (#[a,b]++more).toList.map (·.getId.eraseMacroScopes)
    | _ => []
  direct++binder++tuple++args.toList.flatMap writes

private def bindName (types : Types) (name : Name) (k : Nat) : Types :=
  (name,k)::types.filter (·.1!=name)
private def invalidate (types : Types) (changed : List Name) : Types :=
  types.filter (fun entry=>!changed.contains entry.1)

private partial def lower (types : Types) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | statement::tail =>
    let mut statement:=statement
    if statement.raw.getKind==choiceKind then statement:=⟨statement.raw[0]!⟩
    let mut next:=invalidate types (writes statement.raw)
    match statement with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      next:=bindName types id.getId.eraseMacroScopes (kind types value)
    | `(tritonStmt| for $id:ident in range(0, $stop:tritonExpr) { $inner:tritonStmt* }) =>
      let k:=kind types stop
      let localTypes:=bindName (invalidate types (writes statement.raw)) id.getId.eraseMacroScopes (if k==2 then 2 else 1)
      let inner:=(← lower localTypes inner.toList).toArray
      statement ← if k==2 then
          `(tritonStmt| for $id:ident in range(tl.cast(0, tl.int64), $stop:tritonExpr, 1) { $inner* })
        else `(tritonStmt| for $id:ident in range(0, $stop:tritonExpr) { $inner* })
    | `(tritonStmt| for $id:ident in range($stop:tritonExpr) { $inner:tritonStmt* }) =>
      let k:=kind types stop
      let localTypes:=bindName (invalidate types (writes statement.raw)) id.getId.eraseMacroScopes (if k==2 then 2 else 1)
      let inner:=(← lower localTypes inner.toList).toArray
      statement ← if k==2 then
          `(tritonStmt| for $id:ident in range(tl.cast(0, tl.int64), $stop:tritonExpr, 1) { $inner* })
        else `(tritonStmt| for $id:ident in range($stop:tritonExpr) { $inner* })
    | `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $inner:tritonStmt* }) =>
      let k:=if kind types start==2 || kind types stop==2 then 2 else 1
      let localTypes:=bindName (invalidate types (writes statement.raw)) id.getId.eraseMacroScopes k
      let inner:=(← lower localTypes inner.toList).toArray
      let start ← if k==2 then `(tritonExpr| tl.cast($start:tritonExpr, tl.int64)) else pure start
      statement ← `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $inner* })
    | `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr) { $inner:tritonStmt* }) =>
      let k:=if kind types start==2 || kind types stop==2 then 2 else 1
      let localTypes:=bindName (invalidate types (writes statement.raw)) id.getId.eraseMacroScopes k
      let inner:=(← lower localTypes inner.toList).toArray
      let start ← if k==2 then `(tritonExpr| tl.cast($start:tritonExpr, tl.int64)) else pure start
      statement ← `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr) { $inner* })
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      let localTypes:=invalidate types (writes statement.raw)
      let yes:=(← lower localTypes yes.toList).toArray
      let no:=(← lower localTypes no.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* })
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      let yes:=(← lower (invalidate types (writes statement.raw)) yes.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* })
    | _ => pure ()
    return statement::(← lower next tail)

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "range caller names must form a static list"

private def initialTypes (profile : Syntax) : TermElabM Types := do
  let profile ← elabTermEnsuringType profile (mkConst ``CallSignature)
  let mut types : Types:=[]
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair]) | throwError "scalar name must be literal"
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    let k:=if dtype.isConstOf ``ScalarType.i64 then 2
      else if dtype.isConstOf ``ScalarType.integer || dtype.isConstOf ``ScalarType.i32 then 1 else 0
    types:=bindName types name.toName k
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf name | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then types:=bindName types name.toName 1
  return types

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete caller body"
  let types ← initialTypes stx[1]
  let lowered ← liftMacroM <| lower types statements.toList
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (lowered.map (PythonFullConstructors.render ·.raw))++"\n}")
  NamedFloatDTypeCasts.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.SignedCallRanges
