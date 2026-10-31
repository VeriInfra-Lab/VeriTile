import VeriTile.Triton.DSL.SourceDType
namespace VeriTile.Triton.DSL.RawBodyStores
open Lean

private partial def pointer (aliases : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $name:ident) => aliases.contains name.getId.eraseMacroScopes
  | `(tritonExpr| ($value:tritonExpr)) => pointer aliases value
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr) => pointer aliases base
  | `(tritonExpr| tl.advance($base:tritonExpr, ($_:tritonExpr,*))) => pointer aliases base
  | `(tritonExpr| tl.make_block_ptr($_:ident=$base:tritonExpr,
      $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*),
      $_:ident=($_:tritonExpr,*), $_:ident=($_:num,*))) => pointer aliases base
  | `(tritonExpr| tl.make_block_ptr($_:ident=$base:tritonExpr,
      $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*),
      $_:ident=($_:tritonExpr,*))) => pointer aliases base
  | _ => Bool.false

private partial def hasStore (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal.startsWith "tl.store(") || raw.getArgs.any hasStore

/-- An incoming pointer alias may not acquire a different source type along
one dynamic arm. Fresh aliases are local to that arm/iteration. -/
private partial def rejectRebind (aliases : List Name) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $name:ident = $_:tritonExpr) := raw then
    if aliases.contains name.getId.eraseMacroScopes then
      Macro.throwErrorAt name "pointer alias is rebound under runtime control flow"
  for arg in raw.getArgs do rejectRebind aliases arg

/-- Follow pointer bases, including local block pointers, and lower only the
implicit destination conversion. Unsupported escaping aliases remain untyped
and are rejected by the caller's complete store-coverage check. -/
partial def lower (convert : TSyntax `tritonExpr → MacroM (TSyntax `tritonExpr))
    (aliases : List Name) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | stmt :: tail =>
    let mut stmt := stmt
    let mut next := aliases
    match stmt with
    | `(tritonStmt| $name:ident = $value:tritonExpr) =>
      let key := name.getId.eraseMacroScopes
      next := aliases.filter (· != key)
      if pointer aliases value then next := key::next
    | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
      if pointer aliases destination then
        let value ← convert value
        stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr))
    | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr,
        $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
      if pointer aliases destination then
        let value ← convert value
        stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr,
          $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      for st in yes do rejectRebind aliases st.raw
      for st in no do rejectRebind aliases st.raw
      let yes := (← lower convert aliases yes.toList).toArray
      let no := (← lower convert aliases no.toList).toArray
      stmt ← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* })
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      for st in yes do rejectRebind aliases st.raw
      let yes := (← lower convert aliases yes.toList).toArray
      stmt ← `(tritonStmt| if $condition:tritonExpr { $yes* })
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $inner:tritonStmt* }) =>
      for st in inner do rejectRebind aliases st.raw
      let inner := (← lower convert aliases inner.toList).toArray
      stmt ← `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $inner* })
    | _ =>
      rejectRebind aliases stmt.raw
      if hasStore stmt.raw then Macro.throwErrorAt stmt "unsupported store control flow"
    return stmt::(← lower convert next tail)
end VeriTile.Triton.DSL.RawBodyStores
