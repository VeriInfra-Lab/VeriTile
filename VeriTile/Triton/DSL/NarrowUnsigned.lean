import VeriTile.Triton.DSL.Expansion.Common

/-! Retain byte/halfword arithmetic widths exposed by source casts and aliases.
Triton v3.7.1 keeps a same-kind integer scalar in the tensor's dtype, and emits
integer add/mul at that width (language/semantic.py, computation_type_impl,
add and mul). These implicit modular conversions belong to lowering; the raw
Python quotation receives no extra source statement or cast.

Only statically established unsigned widths are used. Unknown/mixed-kind
expressions retain the ordinary frontend path. -/
namespace VeriTile.Triton.DSL.NarrowUnsigned
open Lean
private abbrev Types := List (Name × Nat)

private def literalFits (width : Nat) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $n:num) => n.getNat < 2^width
  | _ => Bool.false

private def combined (left right : Option Nat) (a b : TSyntax `tritonExpr) : Option Nat :=
  match left,right with
  | some x,some y => some (max x y)
  | some x,none => if literalFits x b then some x else none
  | none,some y => if literalFits y a then some y else none
  | _,_ => none

private def wrap (width : Nat) (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) :=
  if width == 8 then `(tritonExpr| ($e:tritonExpr).to(tl.uint8))
  else `(tritonExpr| ($e:tritonExpr).to(tl.uint16))

private partial def expression (types : Types) (e : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr × Option Nat) := do
  if let some (operand,dtype) := methodCast? e then
    let operand ← if operand.raw.isIdent then do
      let id : Ident := ⟨operand.raw⟩
      `(tritonExpr| $id:ident)
    else pure operand
    let (operand,_) ← expression types operand
    let kind := match dtype with
      | `(tritonDType| tl.uint8) => some 8
      | `(tritonDType| tl.uint16) => some 16
      | _ => none
    return (← `(tritonExpr| ($operand:tritonExpr).to($dtype:tritonDType)),kind)
  let loadedCast ← match e with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kw:tritonMemKwarg]*).to($dtype:tritonDType)) => do
        let load ← `(tritonExpr| tl.load($p:tritonExpr $[, $kw:tritonMemKwarg]*))
        pure (some (load,dtype))
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kw:tritonMemKwarg]*).to($dtype:tritonDType)) => do
        let load ← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kw:tritonMemKwarg]*))
        pure (some (load,dtype))
    | _ => pure none
  if let some (load,dtype) := loadedCast then
    let unsigned := match dtype with
      | `(tritonDType| tl.uint8) | `(tritonDType| tl.uint16) => Bool.true
      | _ => Bool.false
    if unsigned then
      return ← expression types (← `(tritonExpr| ($load:tritonExpr).to($dtype:tritonDType)))
  match e with
  | `(tritonExpr| $id:ident) => return (e,(types.find? (fun p => p.1 == id.getId.eraseMacroScopes)).map Prod.snd)
  | `(tritonExpr| ($inner:tritonExpr)) =>
      let (inner,kind) ← expression types inner
      return (← `(tritonExpr| ($inner:tritonExpr)),kind)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      let (a',ka) ← expression types a
      let (b',kb) ← expression types b
      let e ← `(tritonExpr| $a':tritonExpr + $b':tritonExpr)
      let kind := combined ka kb a b
      return (← (match kind with | some w => wrap w e | none => pure e),kind)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let (a',ka) ← expression types a
      let (b',kb) ← expression types b
      let e ← `(tritonExpr| $a':tritonExpr * $b':tritonExpr)
      let kind := combined ka kb a b
      return (← (match kind with | some w => wrap w e | none => pure e),kind)
  | _ =>
      let args ← e.raw.getArgs.mapM fun raw => do
        return (← expression types (⟨raw⟩ : TSyntax `tritonExpr)).1.raw
      let raw := match e.raw with | .node info kind _ => .node info kind args | _ => e.raw
      return (⟨raw⟩,none)

private def bind (types : Types) (id : Ident) (kind : Option Nat) : Types :=
  let name := id.getId.eraseMacroScopes
  let rest := types.filter (fun p => p.1 != name)
  match kind with | some w => (name,w)::rest | none => rest

private def common (a b : Types) : Types := a.filter (fun p => b.contains p)

private partial def statements (types : Types) (code : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt) × Types) := do
  let mut out := #[]
  let mut types := types
  for st in code do
    match st with
    | `(tritonStmt| $id:ident = $rhs:tritonExpr) =>
        let (rhs,kind) ← expression types rhs
        out := out.push (← `(tritonStmt| $id:ident = $rhs:tritonExpr))
        types := bind types id kind
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        let (condition,_) ← expression types condition
        let (yes,yt) ← statements types yes
        let (no,nt) ← statements types no
        out := out.push (← `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }))
        types := common yt nt
    | `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
        let (condition,_) ← expression types condition
        let (body,bt) ← statements types body
        out := out.push (← `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }))
        types := common types bt
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr)) =>
        let (address,_) ← expression types address
        let (value,_) ← expression types value
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr)))
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $kw:tritonMemKwarg $[, $kws:tritonMemKwarg]*)) =>
        let (address,_) ← expression types address
        let (value,_) ← expression types value
        let kw : TSyntax `tritonMemKwarg := ⟨(← expression types ⟨kw.raw⟩).1.raw⟩
        let kws ← kws.mapM fun k => do
          return (⟨(← expression types ⟨k.raw⟩).1.raw⟩ : TSyntax `tritonMemKwarg)
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $kw:tritonMemKwarg $[, $kws:tritonMemKwarg]*)))
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kws:tritonMemKwarg]*)) =>
        let (address,_) ← expression types address
        let (value,_) ← expression types value
        let (mask,_) ← expression types mask
        let kws ← kws.mapM fun k => do
          return (⟨(← expression types ⟨k.raw⟩).1.raw⟩ : TSyntax `tritonMemKwarg)
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kws:tritonMemKwarg]*)))
    | `(tritonStmt| tl.debug_barrier()) | `(tritonStmt| tl.async_wait()) =>
        out := out.push st
    | _ =>
        out := out.push st
        -- Unsupported statement forms may rebind names. Do not reuse stale
        -- narrow type facts beyond them.
        types := []
  return (out,types)

def lower (code : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt)) :=
  return (← statements [] code).1
end VeriTile.Triton.DSL.NarrowUnsigned
