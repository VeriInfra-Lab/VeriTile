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

private partial def integerLiteral (e : TSyntax `tritonExpr) : Option Int :=
  match e with
  | `(tritonExpr| $n:num) => some n.getNat
  | `(tritonExpr| ($x:tritonExpr))
  | `(tritonExpr| tl.__python_literal($x:tritonExpr)) => integerLiteral x
  | `(tritonExpr| -$x:tritonExpr) => (- ·) <$> integerLiteral x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return (← integerLiteral a) + (← integerLiteral b)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => return (← integerLiteral a) - (← integerLiteral b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => return (← integerLiteral a) * (← integerLiteral b)
  | _ => none

private def scalarFits (width : Nat) (e : TSyntax `tritonExpr) : MacroM Bool := do
  let some n := integerLiteral e | return Bool.false
  unless 0 ≤ n && n < (2 : Int)^width do
    Macro.throwErrorAt e "integer scalar is outside the unsigned tensor's source dtype range"
  return Bool.true

private def combined (left right : Option Nat) (a b : TSyntax `tritonExpr) : MacroM (Option Nat) := do
  match left,right with
  | some x,some y => return some (max x y)
  | some x,none => return if ← scalarFits x b then some x else none
  | none,some y => return if ← scalarFits y a then some y else none
  | _,_ => return none

private partial def pointerWidth (pointers : Types) (e : TSyntax `tritonExpr) : Option Nat :=
  match e with
  | `(tritonExpr| $id:ident) => (pointers.find? (fun p => p.1 == id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| ($x:tritonExpr)) => pointerWidth pointers x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      match pointerWidth pointers a, pointerWidth pointers b with
      | some w,none | none,some w => some w
      | _,_ => none
  | `(tritonExpr| $a:tritonExpr - $_:tritonExpr) => pointerWidth pointers a
  | _ => none

private partial def writtenNames (raw : Syntax) : List Name := Id.run do
  let args := raw.getArgs
  let mut names := []
  if args[0]?.any Syntax.isIdent then
    if let some position := args.findIdx? (fun arg => arg.isAtom &&
        ["=",":=","+=","-=","*=","/="].contains arg.getAtomVal.trimAscii.toString) then
      let rec ids (st : Syntax) : List Name :=
        if st.isIdent then [st.getId.eraseMacroScopes] else st.getArgs.toList.flatMap ids
      names := (args.extract 0 position).toList.flatMap ids
  return names ++ args.toList.flatMap writtenNames

private def wrap (width : Nat) (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) :=
  if width == 8 then `(tritonExpr| ($e:tritonExpr).to(tl.uint8))
  else `(tritonExpr| ($e:tritonExpr).to(tl.uint16))

private partial def expression (types pointers : Types) (e : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr × Option Nat) := do
  if let some (operand,dtype) := methodCast? e then
    let operand ← if operand.raw.isIdent then do
      let id : Ident := ⟨operand.raw⟩
      `(tritonExpr| $id:ident)
    else pure operand
    let (operand,_) ← expression types pointers operand
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
      return ← expression types pointers (← `(tritonExpr| ($load:tritonExpr).to($dtype:tritonDType)))
  match e with
  | `(tritonExpr| $id:ident) => return (e,(types.find? (fun p => p.1 == id.getId.eraseMacroScopes)).map Prod.snd)
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))
  | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      return (e,pointerWidth pointers p)
  | `(tritonExpr| tl.interleave($a:tritonExpr, $b:tritonExpr)) =>
      let (a,ka) ← expression types pointers a
      let (b,kb) ← expression types pointers b
      let kind := if ka == kb then ka else none
      return (← `(tritonExpr| tl.interleave($a:tritonExpr, $b:tritonExpr)),kind)
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) =>
      let (a,ka) ← expression types pointers a
      let (b,_) ← expression types pointers b
      return (← `(tritonExpr| $a:tritonExpr >> $b:tritonExpr),ka)
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr ^ $b:tritonExpr) =>
      let (a',ka) ← expression types pointers a
      let (b',kb) ← expression types pointers b
      let kind ← combined ka kb a b
      let rebuilt ← match e with
        | `(tritonExpr| $_:tritonExpr & $_:tritonExpr) => `(tritonExpr| $a':tritonExpr & $b':tritonExpr)
        | `(tritonExpr| $_:tritonExpr | $_:tritonExpr) => `(tritonExpr| $a':tritonExpr | $b':tritonExpr)
        | _ => `(tritonExpr| $a':tritonExpr ^ $b':tritonExpr)
      return (rebuilt,kind)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      let (a',ka) ← expression types pointers a
      let (b',kb) ← expression types pointers b
      let kind ← combined ka kb a b
      if let some w := kind then
        -- Signed decoding before subtraction avoids Nat's saturating minus;
        -- the final cast implements the source unsigned modular operation.
        let difference ← `(tritonExpr| ($a':tritonExpr).to(tl.int64) - ($b':tritonExpr).to(tl.int64))
        return (← wrap w difference,kind)
      return (← `(tritonExpr| $a':tritonExpr - $b':tritonExpr),none)
  | `(tritonExpr| ($inner:tritonExpr)) =>
      let (inner,kind) ← expression types pointers inner
      return (← `(tritonExpr| ($inner:tritonExpr)),kind)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      let (a',ka) ← expression types pointers a
      let (b',kb) ← expression types pointers b
      let e ← `(tritonExpr| $a':tritonExpr + $b':tritonExpr)
      let kind ← combined ka kb a b
      return (← (match kind with | some w => wrap w e | none => pure e),kind)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let (a',ka) ← expression types pointers a
      let (b',kb) ← expression types pointers b
      let e ← `(tritonExpr| $a':tritonExpr * $b':tritonExpr)
      let kind ← combined ka kb a b
      return (← (match kind with | some w => wrap w e | none => pure e),kind)
  | _ =>
      let args ← e.raw.getArgs.mapM fun raw => do
        return (← expression types pointers (⟨raw⟩ : TSyntax `tritonExpr)).1.raw
      let raw := match e.raw with | .node info kind _ => .node info kind args | _ => e.raw
      return (⟨raw⟩,none)

private def bind (types : Types) (id : Ident) (kind : Option Nat) : Types :=
  let name := id.getId.eraseMacroScopes
  let rest := types.filter (fun p => p.1 != name)
  match kind with | some w => (name,w)::rest | none => rest

private def common (a b : Types) : Types := a.filter (fun p => b.contains p)

private partial def statements (types pointers : Types) (code : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt) × Types × Types) := do
  let mut out := #[]
  let mut types := types
  let mut pointers := pointers
  for st in code do
    match st with
    | `(tritonStmt| $id:ident = $rhs:tritonExpr) =>
        let pointed := pointerWidth pointers rhs
        let (rhs,kind) ← expression types pointers rhs
        out := out.push (← `(tritonStmt| $id:ident = $rhs:tritonExpr))
        types := bind types id kind
        pointers := bind pointers id pointed
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        let (condition,_) ← expression types pointers condition
        let (yes,yt,yp) ← statements types pointers yes
        let (no,nt,np) ← statements types pointers no
        out := out.push (← `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }))
        types := common yt nt
        pointers := common yp np
    | `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }) =>
        let (condition,_) ← expression types pointers condition
        let (body,bt,bp) ← statements types pointers body
        out := out.push (← `(tritonStmt| if $condition:tritonExpr { $body:tritonStmt* }))
        types := common types bt
        pointers := common pointers bp
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr)) =>
        let (address,_) ← expression types pointers address
        let (value,_) ← expression types pointers value
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr)))
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $kw:tritonMemKwarg $[, $kws:tritonMemKwarg]*)) =>
        let (address,_) ← expression types pointers address
        let (value,_) ← expression types pointers value
        let kw : TSyntax `tritonMemKwarg := ⟨(← expression types pointers ⟨kw.raw⟩).1.raw⟩
        let kws ← kws.mapM fun k => do
          return (⟨(← expression types pointers ⟨k.raw⟩).1.raw⟩ : TSyntax `tritonMemKwarg)
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $kw:tritonMemKwarg $[, $kws:tritonMemKwarg]*)))
    | `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kws:tritonMemKwarg]*)) =>
        let (address,_) ← expression types pointers address
        let (value,_) ← expression types pointers value
        let (mask,_) ← expression types pointers mask
        let kws ← kws.mapM fun k => do
          return (⟨(← expression types pointers ⟨k.raw⟩).1.raw⟩ : TSyntax `tritonMemKwarg)
        out := out.push (← `(tritonStmt| tl.store($address:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kws:tritonMemKwarg]*)))
    | `(tritonStmt| tl.debug_barrier()) | `(tritonStmt| tl.async_wait()) =>
        out := out.push st
    | _ =>
        out := out.push st
        -- Unsupported statement forms may rebind names. Do not reuse stale
        -- narrow type facts beyond them.
        types := []
        pointers := pointers.filter (fun p => !(writtenNames st.raw).contains p.1)
  return (out,types,pointers)

def lower (code : Array (TSyntax `tritonStmt)) (pointers : List (Name × Nat) := []) : MacroM (Array (TSyntax `tritonStmt)) :=
  return (← statements [] pointers code).1
end VeriTile.Triton.DSL.NarrowUnsigned
