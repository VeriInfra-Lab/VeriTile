import VeriTile.Triton.DSL.SingleAssignmentPointerOffsets

/-! Keep already signed integer address aliases signed when later memory uses
would otherwise pin their definitions to Nat. Only expressions known to have
signed integer type from the source casts, integer operands and signed loop
counter receive the internal identity cast; floating and unknown expressions
remain unchanged. All source assignments and operations are retained. -/
namespace VeriTile.Triton.DSL.SignedIntegerAddressAliases
open Lean Elab Term Meta
private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "address alias profile must have statically known names"
private def stringValue (e : Expr) : TermElabM String := do
  let .lit (.strVal s) ← whnf e | throwError "address alias name must be literal"
  return s
private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers
private partial def addresses (raw : Syntax) : List Name :=
  let direct := match raw with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) => identifiers p.raw
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => identifiers p.raw
    | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*).to($_:tritonDType)) => identifiers p.raw
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($_:tritonDType)) => identifiers p.raw
    | `(tritonStmt| tl.store($p:tritonExpr, $_:tritonExpr)) => identifiers p.raw
    | `(tritonStmt| tl.store($p:tritonExpr, $_:tritonExpr, $_:tritonMemKwarg $[, $_:tritonMemKwarg]*)) => identifiers p.raw
    | `(tritonStmt| tl.store($p:tritonExpr, $_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => identifiers p.raw
    | _ => []
  direct ++ raw.getArgs.toList.flatMap addresses

/-- 0: unknown/floating; 1: integer with no signed64 evidence; 2: signed64. -/
partial def kind (names : List (Name × Nat)) (e : TSyntax `tritonExpr) : Nat :=
  if let some (_,dt):=methodCast? e then
    match dt with | `(tritonDType| tl.int64) => 2 | _ => 0
  else match e with
  | `(tritonExpr| $id:ident) => (names.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd |>.getD 0
  | `(tritonExpr| $n:num) => if n.getNat<2^63 then 1 else 0
  | `(tritonExpr| ($inner:tritonExpr)) => kind names inner
  | `(tritonExpr| tl.cast($_:tritonExpr, tl.int64)) => 2
  | `(tritonExpr| tl.full((), $_:tritonExpr, tl.int64)) => 2
  | `(tritonExpr| tl.full([], $_:tritonExpr, tl.int64)) => 2
  | `(tritonExpr| tl.full([], $_:tritonExpr, dtype=tl.int64)) => 2
  | `(tritonExpr| tl.program_id($_:tritonExpr))
  | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr)) => 1
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
    let ka:=kind names a; let kb:=kind names b
    if ka==0 || kb==0 then 0 else max ka kb
  | `(tritonExpr| tl.where($_:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
    let ka:=kind names a; let kb:=kind names b
    if ka==0 || kb==0 then 0 else max ka kb
  | _ => 0

def bindName (names : List (Name × Nat)) (name : Name) (value : Nat) :=
  (name,value)::names.filter (·.1!=name)
private partial def lowerStatements (used : List Name) (names : List (Name × Nat))
    (stmts : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt) × List (Name × Nat)) := do
  let mut out:=[];let mut names:=names
  for stmt in stmts do
    match stmt with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      let k:=kind names value
      let next ← if k==2 && used.contains id.getId.eraseMacroScopes then
          `(tritonStmt| $id:ident = tl.cast($value:tritonExpr, tl.int64))
        else pure stmt
      out:=out++[next];names:=bindName names id.getId.eraseMacroScopes k
    | `(tritonStmt| $id:ident += $value:tritonExpr) =>
      let old:=(names.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd |>.getD 0
      let k:=kind names value
      names:=bindName names id.getId.eraseMacroScopes (if old>0 && k>0 then max old k else 0)
      out:=out++[stmt]
    | `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
      let signed:=kind names start==2 || kind names stop==2
      let (body,_) ← lowerStatements used (bindName names id.getId.eraseMacroScopes (if signed then 2 else 1)) body.toList
      let body:=body.toArray
      out:=out++[← `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* })]
      -- Body writes may rebind registers; retain no stale alias evidence.
      names:=names.map fun p=>if (identifiers stmt.raw).contains p.1 then (p.1,0) else p
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      let (yes,_) ← lowerStatements used names yes.toList
      let (no,_) ← lowerStatements used names no.toList
      let yes:=yes.toArray;let no:=no.toArray
      out:=out++[← `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })]
      names:=names.map fun p=>if (identifiers stmt.raw).contains p.1 then (p.1,0) else p
    | `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
      let (body,_) ← lowerStatements used names body.toList
      let body:=body.toArray
      out:=out++[← `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* })]
      names:=names.map fun p=>if (identifiers stmt.raw).contains p.1 then (p.1,0) else p
    | _ =>
      out:=out++[stmt]
      names:=names.map fun p=>if (identifiers stmt.raw).contains p.1 then (p.1,0) else p
  return (out,names)

/-- Source integer evidence before the first statement. -/
def initialKinds (profile : Expr) : TermElabM (List (Name × Nat)) := do
  let mut names : List (Name × Nat):=[]
  for pair in ← listValues (← mkAppM ``Triton.CallSignature.scalars #[profile]) do
    let name ← stringValue (← mkAppM ``Prod.fst #[pair])
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    if dtype.isConstOf ``Triton.ScalarType.i64 then
      names:=bindName names (Name.mkSimple name) 2
    else if dtype.isConstOf ``Triton.ScalarType.integer || dtype.isConstOf ``Triton.ScalarType.i32 then
      names:=bindName names (Name.mkSimple name) 1
  for constant in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let name ← stringValue constant
    let value ← elabTerm (mkIdent (Name.mkSimple name)) none
    if (← whnf (← inferType value)).isConstOf ``Nat then names:=bindName names (Name.mkSimple name) 1
  return names

def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let names ← initialKinds profile
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let (stmts,_) ← liftMacroM (lowerStatements (addresses parsed) names stmts.toList)
  let source:="{\n"++String.intercalate "\n" (stmts.map (PythonFullConstructors.render ·.raw))++"\n}"
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))
end VeriTile.Triton.DSL.SignedIntegerAddressAliases
