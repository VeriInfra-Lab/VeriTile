import VeriTile.Triton.DSL.PythonFullConstructors

/-! Storage-polymorphic loads keep their typed payload. When an immediately
following addition consumes a known FP32 load, enter the existing FP32 compute
carrier explicitly. The source quotation stays unchanged. This is a type
conversion at the arithmetic operand, never a floating identity assumption. -/
namespace VeriTile.Triton.DSL.FP32LoadedUpdates
open Lean Elab Term Meta

private partial def values (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]!::(← values value.getAppArgs[2]!)
  throwError "pointer metadata must be static"

private partial def pointer (names : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $name:ident)=>names.contains name.getId.eraseMacroScopes
  | `(tritonExpr| ($value:tritonExpr))=>pointer names value
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr)=>pointer names base
  | `(tritonExpr| tl.advance($base:tritonExpr, ($_:tritonExpr,*)))=>pointer names base
  | `(tritonExpr| tl.make_block_ptr($_:ident=$base:tritonExpr,
      $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*), $_:ident=($_:tritonExpr,*),
      $_:ident=($_:tritonExpr,*), $_:ident=($_:num,*)))=>pointer names base
  | _=>Bool.false

private partial def lower (names : List Name) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | []=>return []
  | st::rest=>
    let mut next:=names
    let mut st:=st
    let mut rest:=rest
    if let `(tritonStmt| $name:ident = $value:tritonExpr):=st then
      next:=names.filter (· != name.getId.eraseMacroScopes)
      if pointer names value then next:=name.getId.eraseMacroScopes::next
      if let `(tritonExpr| tl.load($source:tritonExpr $[, $_:tritonMemKwarg]*)):=value then
        if pointer names source then
          if let update::tail:=rest then
            if let `(tritonStmt| $lhs:ident += $rhs:tritonExpr):=update then
              if lhs.getId.eraseMacroScopes==name.getId.eraseMacroScopes then
                let update ← `(tritonStmt| $lhs:ident = ($lhs:ident).to(tl.float32) + $rhs:tritonExpr)
                rest:=update::tail
    match st with
    | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $inside:tritonStmt* })=>
      let inside:=(← lower names inside.toList).toArray
      st ← `(tritonStmt| for $counter:ident in range($lo, $hi, $stride) { $inside* })
    | `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr) { $inside:tritonStmt* })=>
      let inside:=(← lower names inside.toList).toArray
      st ← `(tritonStmt| for $counter:ident in range(0, $hi) { $inside* })
    | _=>pure ()
    return st::(← lower next rest)

def prepare (call : Syntax) (body : Syntax) : TermElabM Syntax := do
  let profile ← elabTermEnsuringType call[1] (mkConst ``CallSignature)
  let mut names:List Name:=[]
  for item in ← values (← mkAppM ``CallSignature.floatingPointers #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[item])
    if dtype.isConstOf ``FloatDType.fp32 then
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[item]) | throwError "literal pointer name required"
      names:=names++[name.toName]
  let `(triton { $statements:tritonStmt* }):=body | throwError "complete source required"
  let statements:=(← liftMacroM (lower names statements.toList)).toArray
  return (← `(triton { $statements* })).raw
end VeriTile.Triton.DSL.FP32LoadedUpdates
