import VeriTile.Triton.DSL.TypedStores

/-! The caller may retain FP32 memory as a typed channel. A value reduction
must then project its operand to the FP32 arithmetic carrier. Track actual
storage reads and copies: ordinary arithmetic registers never acquire an
extra identity cast, and raw Python bodies remain untouched. -/
namespace VeriTile.Triton.DSL.TypedStorageReduction
open Lean

private def identName (id : Ident) := id.getId.eraseMacroScopes
private partial def pointer (names : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $id:ident) => names.contains (identName id)
  | `(tritonExpr| ($e:tritonExpr)) => pointer names e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => pointer names a != pointer names b
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => pointer names a && !pointer names b
  | _ => Bool.false

private partial def overridesDType (raw : Syntax) : Bool :=
  (raw.isIdent && raw.getId.eraseMacroScopes == `dtype) ||
  (raw.isAtom && raw.getAtomVal == "dtype") || raw.getArgs.any overridesDType

private partial def carrier (pointers values : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $id:ident) => values.contains (identName id)
  | `(tritonExpr| ($e:tritonExpr))
  | `(tritonExpr| $e:tritonExpr[:, None])
  | `(tritonExpr| $e:tritonExpr[None, :]) => carrier pointers values e
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))
  | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      pointer pointers p && !kwargs.any (fun kw => overridesDType kw.raw)
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      carrier pointers values e && !kwargs.any (fun kw => overridesDType kw.raw)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) =>
      carrier pointers values a && carrier pointers values b
  | _ => Bool.false

private partial def expression (pointers values : List Name) (raw : Syntax) : MacroM Syntax := do
  let args ← raw.getArgs.mapM (expression pointers values)
  let raw := match raw with | .node info kind _ => Syntax.node info kind args | _ => raw
  let project (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
    if carrier pointers values e then `(tritonExpr| ($e:tritonExpr).to(tl.float32)) else pure e
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      if carrier pointers values a == carrier pointers values b then return raw
      let a ← project a
      let b ← project b
      return (← `(tritonExpr| $a:tritonExpr + $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      if carrier pointers values a == carrier pointers values b then return raw
      let a ← project a
      let b ← project b
      return (← `(tritonExpr| $a:tritonExpr - $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      if carrier pointers values a == carrier pointers values b then return raw
      let a ← project a
      let b ← project b
      return (← `(tritonExpr| $a:tritonExpr * $b:tritonExpr)).raw
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) =>
      if carrier pointers values a == carrier pointers values b then return raw
      let a ← project a
      let b ← project b
      return (← `(tritonExpr| $a:tritonExpr / $b:tritonExpr)).raw
  | `(tritonExpr| tl.max($e:tritonExpr, $axisIndex:num)) =>
      let e ← project e
      return (← `(tritonExpr| tl.max($e:tritonExpr, axis = $axisIndex:num))).raw
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      let e ← project e
      return (← `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*))).raw
  | _ => return raw

private def common (a b : List Name) := a.filter b.contains

private partial def block (pointers values : List Name) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt) × List Name × List Name) := do
  match body with
  | [] => return ([],pointers,values)
  | stmt :: rest =>
      let mut stmt := stmt
      let mut nextPointers := pointers
      let mut nextValues := values
      let loop (_id : Ident) (body : Array (TSyntax `tritonStmt)) := do
        -- Incoming registers can change representation between iterations.
        -- Only fresh reads/copies inside this loop establish provenance.
        let (body,p,_) ← block pointers [] body.toList
        return (body.toArray, common pointers p, ([] : List Name))
      match stmt with
      | `(tritonStmt| $id:ident = $rhs:tritonExpr) =>
          let lowered : TSyntax `tritonExpr := ⟨← expression pointers values rhs.raw⟩
          stmt ← `(tritonStmt| $id:ident = $lowered:tritonExpr)
          nextPointers := pointers.filter (· != identName id)
          nextValues := values.filter (· != identName id)
          if pointer pointers rhs then nextPointers := identName id :: nextPointers
          if carrier pointers values rhs then nextValues := identName id :: nextValues
      | `(tritonStmt| $id:ident = tl.max($rhs:tritonExpr, $axisIndex:num)) =>
          let e ← `(tritonExpr| tl.max($rhs:tritonExpr, axis = $axisIndex:num))
          let e : TSyntax `tritonExpr := ⟨← expression pointers values e.raw⟩
          stmt ← `(tritonStmt| $id:ident = $e:tritonExpr)
          nextPointers := pointers.filter (· != identName id)
          nextValues := values.filter (· != identName id)
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          let (yes,yp,yv) ← block pointers values yes.toList
          let (no,np,nv) ← block pointers values no.toList
          let yes := yes.toArray; let no := no.toArray
          stmt ← `(tritonStmt| if $c { $yes* } else { $no* })
          nextPointers := common yp np; nextValues := common yv nv
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) =>
          let (yes,p,v) ← block pointers values yes.toList
          let yes := yes.toArray
          stmt ← `(tritonStmt| if $c { $yes* })
          nextPointers := common pointers p; nextValues := common values v
      | `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) =>
          let (body,p,v) ← loop id body
          stmt ← `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body* })
          nextPointers := p; nextValues := v
      | `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr) { $body:tritonStmt* }) =>
          let (body,p,v) ← loop id body
          stmt ← `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr) { $body* })
          nextPointers := p; nextValues := v
      | `(tritonStmt| for $id:ident in range($n:tritonExpr) { $body:tritonStmt* }) =>
          let (body,p,v) ← loop id body
          stmt ← `(tritonStmt| for $id:ident in range($n:tritonExpr) { $body* })
          nextPointers := p; nextValues := v
      | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr))
      | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr, $_:tritonMemKwarg $[, $_:tritonMemKwarg]*))
      | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) =>
          stmt := ⟨← expression pointers values stmt.raw⟩
      | `(tritonStmt| $id:ident += $_:tritonExpr)
      | `(tritonStmt| $id:ident -= $_:tritonExpr)
      | `(tritonStmt| $id:ident *= $_:tritonExpr)
      | `(tritonStmt| $id:ident /= $_:tritonExpr) =>
          nextValues := values.filter (· != identName id)
      | `(tritonStmt| tl.debug_barrier()) => pure ()
      | _ =>
          -- Unknown assignments may invalidate any tracked alias or value.
          nextValues := []
          nextPointers := []
      let (rest,p,v) ← block nextPointers nextValues rest
      return (stmt :: rest,p,v)

def lower (typedFP32Pointers : List Name) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  if typedFP32Pointers.isEmpty then return body
  return (← block typedFP32Pointers [] body).1
end VeriTile.Triton.DSL.TypedStorageReduction
