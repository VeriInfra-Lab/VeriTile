import VeriTile.Triton.DSL.Syntax
import Lean

/-! Keep a floating load as a complete compute expression when composing it
with arithmetic. The inserted registers are compiler temporaries; source text
and source evaluation order remain unchanged. A conditional assignment keeps its loads inside the selected branch.
Loads inside conditions and other unsupported expression positions fail closed. -/
namespace VeriTile.Triton.DSL.NestedLoadAssignments
open Lean

private partial def namesIn (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap namesIn

private def floatingLoad (pointers : List Name) (e : TSyntax `tritonExpr) : Bool :=
  let address : Option (TSyntax `tritonExpr) := match e with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
    | _ => none
  address.any (fun p => (namesIn p.raw).any pointers.contains)

private partial def hasFloatingLoad (pointers : List Name) (raw : Syntax) : Bool :=
  let e : TSyntax `tritonExpr := ⟨raw⟩
  let castLoad := match e with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*).to($_:tritonDType)) =>
        (namesIn p.raw).any pointers.contains
    | _ => Bool.false
  castLoad || floatingLoad pointers e || raw.getArgs.any (hasFloatingLoad pointers)

abbrev LowerM := StateT (List Name) MacroM
private def fresh : LowerM Ident := do
  let used ← get
  let mut index := 0
  while used.contains (Name.mkSimple ("__veritile_source_load_" ++ toString index)) do
    index := index + 1
  let name := Name.mkSimple ("__veritile_source_load_" ++ toString index)
  set (name :: used)
  return mkIdent name

private def supportedCast (dt : TSyntax `tritonDType) : Bool :=
  match dt with
  | `(tritonDType| tl.float32) | `(tritonDType| tl.float64)
  | `(tritonDType| tl.float16) | `(tritonDType| tl.bfloat16) => Bool.true
  | _ => Bool.false

/-- Only eager arithmetic operands are lifted. Left operands are evaluated
before right operands; explicit casts stay at their original expression site. -/
private partial def operands (pointers : List Name) (e : TSyntax `tritonExpr) :
    LowerM (Array (TSyntax `tritonStmt) × TSyntax `tritonExpr) := do
  unless hasFloatingLoad pointers e.raw do return (#[],e)
  if floatingLoad pointers e then
    let id ← fresh
    return (#[← `(tritonStmt| $id:ident = $e:tritonExpr)], ← `(tritonExpr| $id:ident))
  match e with
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      let (sa,a) ← operands pointers a; let (sb,b) ← operands pointers b
      return (sa ++ sb, ← `(tritonExpr| $a:tritonExpr + $b:tritonExpr))
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      let (sa,a) ← operands pointers a; let (sb,b) ← operands pointers b
      return (sa ++ sb, ← `(tritonExpr| $a:tritonExpr - $b:tritonExpr))
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let (sa,a) ← operands pointers a; let (sb,b) ← operands pointers b
      return (sa ++ sb, ← `(tritonExpr| $a:tritonExpr * $b:tritonExpr))
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) =>
      let (sa,a) ← operands pointers a; let (sb,b) ← operands pointers b
      return (sa ++ sb, ← `(tritonExpr| $a:tritonExpr / $b:tritonExpr))
  | `(tritonExpr| -$a:tritonExpr) =>
      let (sa,a) ← operands pointers a
      return (sa, ← `(tritonExpr| -$a:tritonExpr))
  | `(tritonExpr| ($a:tritonExpr)) =>
      let (sa,a) ← operands pointers a
      return (sa, ← `(tritonExpr| ($a:tritonExpr)))
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
      unless supportedCast dt do
        liftM <| Macro.throwErrorAt e "nested load source precision is unsupported by the current frontend"
      let loaded ← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))
      let (sa,a) ← operands pointers loaded
      return (sa, ← `(tritonExpr| ($a:tritonExpr).to($dt:tritonDType)))
  | `(tritonExpr| ($a:tritonExpr).to($dt:tritonDType)) =>
      unless supportedCast dt do
        liftM <| Macro.throwErrorAt e "nested load source precision is unsupported by the current frontend"
      let (sa,a) ← operands pointers a
      return (sa, ← `(tritonExpr| ($a:tritonExpr).to($dt:tritonDType)))
  | _ =>
      if hasFloatingLoad pointers e.raw then
        liftM <| Macro.throwErrorAt e "nested floating load requires supported eager arithmetic operands"
      return (#[],e)

private def arithmeticRoot (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $_:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr - $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr * $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr / $_:tritonExpr) => Bool.true
  | _ => Bool.false

private partial def block (pointers : List Name) (body : Array (TSyntax `tritonStmt)) :
    LowerM (Array (TSyntax `tritonStmt)) := do
  let mut result := #[]
  for st in body do
    match st with
    | `(tritonStmt| $id:ident = $rhs:tritonExpr) =>
        if let `(tritonExpr| ($yes:tritonExpr if $condition:tritonExpr else $no:tritonExpr)) := rhs then
          if hasFloatingLoad pointers rhs.raw then
            let yes ← block pointers #[← `(tritonStmt| $id:ident = $yes:tritonExpr)]
            let no ← block pointers #[← `(tritonStmt| $id:ident = $no:tritonExpr)]
            result := result.push (← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* }))
            continue
        if arithmeticRoot rhs && hasFloatingLoad pointers rhs.raw then
          let (prelude,rhs) ← operands pointers rhs
          result := result ++ prelude
          result := result.push (← `(tritonStmt| $id:ident = $rhs:tritonExpr))
        else result := result.push st
    | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        let yes ← block pointers yes; let no ← block pointers no
        result := result.push (← `(tritonStmt| if $c:tritonExpr { $yes* } else { $no* }))
    | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) =>
        let yes ← block pointers yes
        result := result.push (← `(tritonStmt| if $c:tritonExpr { $yes* }))
    | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
        let body ← block pointers body
        result := result.push (← `(tritonStmt| for $i:ident in range($start,$stop,$step) { $body* }))
    | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
        let body ← block pointers body
        result := result.push (← `(tritonStmt| for $i:ident in range($start,$stop) { $body* }))
    | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $body:tritonStmt* }) =>
        let body ← block pointers body
        result := result.push (← `(tritonStmt| for $i:ident in range($stop) { $body* }))
    | _ => result := result.push st
  return result

def lower (pointers : List Name) (body : Array (TSyntax `tritonStmt))
    (reserved : List Name := []) : MacroM (Array (TSyntax `tritonStmt)) := do
  let used := reserved ++ pointers ++ body.toList.flatMap (fun s => namesIn s.raw)
  return (← (block pointers body).run used).1
end VeriTile.Triton.DSL.NestedLoadAssignments
