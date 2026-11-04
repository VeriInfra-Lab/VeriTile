import VeriTile.Triton.DSL.PythonFullConstructors

/-! Compile-time natural equalities are checked against immutable caller
constexprs before static assertions are removed from executable statements.
Unproved or runtime-dependent assertions fail elaboration. -/
namespace VeriTile.Triton.DSL.VerifiedStaticAssertions
open Lean Elab Term Meta
private partial def values (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then return value.getAppArgs[1]!::(← values value.getAppArgs[2]!)
  throwError "static assertions require literal constexpr names"
private partial def natural (names : List Name) (value : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match value with
  | `(tritonExpr| $n:num)=>`(($n:num : Nat))
  | `(tritonExpr| $name:ident)=>
    if names.contains name.getId.eraseMacroScopes then `(($name:ident : Nat))
    else Macro.throwError "static assertion depends on a non-constexpr name"
  | `(tritonExpr| ($value:tritonExpr))=>natural names value
  | _=>Macro.throwError "static assertion requires a natural constexpr equality"
private partial def immutable (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  if raw.getKind.getString!.startsWith "tritonStmt" then
    for operator in ["=",":=","+=","-=","*=","/=","//=","%="] do
      if let some i:=args.findIdx? (fun arg=>arg.isAtom && arg.getAtomVal.trimAscii.toString==operator) then
        if (args.take i).any (fun arg=>arg.isIdent && names.contains arg.getId.eraseMacroScopes) then
          throwError "asserted constexprs must remain immutable"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwError "a loop cannot shadow an asserted constexpr"
  for child in raw.getArgs do immutable names child
def prepare (call : Syntax) (body : Syntax) : TermElabM Syntax := do
  let profile ← elabTermEnsuringType call[1] (mkConst ``CallSignature)
  let names ← (← values (← mkAppM ``CallSignature.constexpr #[profile])).mapM fun value=>do
    let .lit (.strVal name) ← whnf value | throwError "constexpr names must be literal"
    return name.toName
  let `(triton { $statements:tritonStmt* }):=body | throwError "complete source required"
  let mut result:=#[]
  for statement in statements do
    if let `(tritonStmt| tl.static_assert($condition:tritonExpr)):=statement then
      let `(tritonExpr| $left:tritonExpr == $right:tritonExpr):=condition
        | throwError "static assertion requires an equality"
      immutable names body
      let left ← elabTermEnsuringType (← liftMacroM (natural names left)) (mkConst ``Nat)
      let right ← elabTermEnsuringType (← liftMacroM (natural names right)) (mkConst ``Nat)
      unless ← isDefEq left right do throwError "static assertion is not established by the caller"
      continue
    result:=result.push statement
  return (← `(triton { $result* })).raw
end VeriTile.Triton.DSL.VerifiedStaticAssertions
