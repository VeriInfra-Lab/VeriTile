import VeriTile.Triton.DSL.SourceDType
import Lean

/-! Preserve typed load operations inside store expressions. The statement IR
represents a typed load as a ComputeExpr, so a load nested in an Op reduction
must first receive a private register. This is frontend lowering of the pasted
source, not another task implementation. Branches and iterations keep their
loads inside their original control-flow scope. -/
namespace VeriTile.Triton.DSL.RawBodyNestedLoads
open Lean

private partial def names (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap names

private partial def containsLoad (raw : Syntax) : Bool :=
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  (match expression with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | _ => Bool.false) || raw.getArgs.any containsLoad

private partial def value (used : List Name) (next : Nat) (raw : Syntax) :
    MacroM (List (TSyntax `tritonStmt) × Syntax × Nat) := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let conditional := match expression with
    | `(tritonExpr| ($_:tritonExpr if $_:tritonExpr else $_:tritonExpr)) => Bool.true
    | `(tritonExpr| $_:tritonExpr and $_:tritonExpr) => Bool.true
    | `(tritonExpr| $_:tritonExpr or $_:tritonExpr) => Bool.true
    | _ => Bool.false
  if conditional && containsLoad raw then
    Macro.throwErrorAt raw "nested loads under short-circuit expressions require an explicit control-flow lowering"
  let mut prelude := []
  let mut next := next
  let mut args := #[]
  for arg in raw.getArgs do
    let (before,arg,after) ← value used next arg
    prelude := prelude ++ before
    args := args.push arg
    next := after
  let raw := raw.setArgs args
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let loaded := match expression with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
    | _ => Bool.false
  unless loaded do return (prelude,raw,next)
  let mut name := Name.mkSimple ("__veritile_load_" ++ toString next)
  while used.contains name do
    next := next+1
    name := Name.mkSimple ("__veritile_load_" ++ toString next)
  let id := mkIdent name
  let stmt ← `(tritonStmt| $id:ident = $expression:tritonExpr)
  let reference ← `(tritonExpr| $id:ident)
  return (prelude ++ [stmt],reference.raw,next+1)

private partial def body (used : List Name) (next : Nat) (stmts : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt) × Nat) := do
  let mut result := []
  let mut next := next
  for stmt in stmts do
    let mut prelude := []
    let stmt ← match stmt with
      | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr)) => do
          let (before,v,after) ← value used next v.raw
          prelude := before
          next := after
          let v : TSyntax `tritonExpr := ⟨v⟩
          `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr))
      | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr,
          $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) => do
          let (before,v,after) ← value used next v.raw
          prelude := before
          next := after
          let v : TSyntax `tritonExpr := ⟨v⟩
          `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr,
            $kw:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr,
          $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
          let (before,v,after) ← value used next v.raw
          prelude := before
          next := after
          let v : TSyntax `tritonExpr := ⟨v⟩
          `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr,
            $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => do
          let (yes,after) ← body used next yes.toList
          let (no,after) ← body used after no.toList
          next := after
          let yes := yes.toArray
          let no := no.toArray
          `(tritonStmt| if $c:tritonExpr { $yes* } else { $no* })
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) => do
          let (yes,after) ← body used next yes.toList
          next := after
          let yes := yes.toArray
          `(tritonStmt| if $c:tritonExpr { $yes* })
      | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $s:tritonExpr) { $inner:tritonStmt* }) => do
          let (inner,after) ← body used next inner.toList
          next := after
          let inner := inner.toArray
          `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $s:tritonExpr) { $inner* })
      | _ => pure stmt
    result := result ++ prelude ++ [stmt]
  return (result,next)

def lower (stmts : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  return (← body (stmts.flatMap (fun st => names st.raw)) 0 stmts).1

end VeriTile.Triton.DSL.RawBodyNestedLoads
