import VeriTile.Triton.DSL.Syntax

/-! Preserve literal integer tensor steps in Python range calls. A preceding
scalar assignment remains in the program; only a proven syntactically constant
captured bound is materialized. Unknown control flow drops all value facts.
No runtime signed value is truncated to Nat, and zero steps are rejected. -/
namespace VeriTile.Triton.DSL.KnownRangeStep
open Lean
private abbrev Known := List (Name × Int)
private def key (id : Ident) := id.getId.eraseMacroScopes
private def forget (known : Known) (id : Ident) : Known := known.filter (·.1 != key id)

private partial def literal (known : Known) : TSyntax `tritonExpr → Option Int
  | `(tritonExpr| tl.__python_literal($e:tritonExpr)) => literal known e
  | `(tritonExpr| ($e:tritonExpr)) => literal known e
  | `(tritonExpr| $n:num) => some n.getNat
  | `(tritonExpr| -$e:tritonExpr) => (- ·) <$> literal [] e
  | `(tritonExpr| $id:ident) => (known.find? (·.1 == key id)).map Prod.snd
  | _ => none

private def signedFull (known : Known) (value : TSyntax `tritonExpr)
    (dtype : TSyntax `tritonDType) : Option Int := do
  let width ← match dtype with
    | `(tritonDType| tl.int8) => some 8
    | `(tritonDType| tl.int16) => some 16
    | `(tritonDType| tl.int32) => some 32
    | `(tritonDType| tl.int64) => some 64
    | _ => none
  let n ← literal known value
  if -(2^(width-1) : Int) ≤ n && n < 2^(width-1) then some n else none

private def assigned (known : Known) (value : TSyntax `tritonExpr) : Option Int :=
  match value with
  | `(tritonExpr| tl.full([], $value:tritonExpr, $dtype:tritonDType)) =>
      signedFull known value dtype
  | `(tritonExpr| tl.full([], $value:tritonExpr, $name:ident=$dtype:tritonDType)) =>
      if name.getId == `dtype then signedFull known value dtype else none
  | _ => do
      let n ← literal known value
      -- Keep ordinary tensor-literal inference inside the signed64 domain.
      if -(2^63 : Int) ≤ n && n < 2^63 then some n else none

private def bind (known : Known) (id : Ident) (value : Option Int) : Known :=
  let rest := forget known id
  match value with | some n => (key id,n) :: rest | none => rest

private def stepExpr (value : Int) : MacroM (TSyntax `tritonExpr) := do
  if value == 0 then Macro.throwError "range step cannot be zero"
  let n : TSyntax `num := ⟨Syntax.mkNumLit (toString value.natAbs)⟩
  if value < 0 then `(tritonExpr| -$n:num) else `(tritonExpr| $n:num)

private partial def lowerStmts (known : Known) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | st :: tail =>
    let (st, next) ← match st with
      | `(tritonStmt| $id:ident = $value:tritonExpr) =>
          pure (st, bind known id (assigned known value))
      | `(tritonStmt| $id:ident := $value:tritonExpr) =>
          pure (st, bind known id (assigned known value))
      | `(tritonStmt| $id:ident += $_:tritonExpr)
      | `(tritonStmt| $id:ident -= $_:tritonExpr)
      | `(tritonStmt| $id:ident *= $_:tritonExpr)
      | `(tritonStmt| $id:ident /= $_:tritonExpr) => pure (st, forget known id)
      | `(tritonStmt| while $condition:tritonExpr { $nested:tritonStmt* }) => do
          let nested := (← lowerStmts [] nested.toList).toArray
          pure (← `(tritonStmt| while $condition { $nested* }), [])
      | `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* }) => do
          let stride ← match literal known stride with
            | some n => stepExpr n
            | none => pure stride
          -- No entry fact is assumed invariant across an arbitrary loop body.
          let nested := (← lowerStmts [] nested.toList).toArray
          let rewritten ← `(tritonStmt| for $id:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* })
          pure (rewritten, [])
      | `(tritonStmt| tl.store($_p:tritonExpr, $_v:tritonExpr)) => pure (st, known)
      | `(tritonStmt| tl.store($_p:tritonExpr, $_v:tritonExpr, $_kw:tritonMemKwarg $[, $_kws:tritonMemKwarg]*)) => pure (st, known)
      | _ => pure (st, [])
    return st :: (← lowerStmts next tail)

def lower (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt)) :=
  return (← lowerStmts [] body.toList).toArray
end VeriTile.Triton.DSL.KnownRangeStep
