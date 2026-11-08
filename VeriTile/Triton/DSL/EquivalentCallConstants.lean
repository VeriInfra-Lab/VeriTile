import VeriTile.Triton.DSL.PythonFullConstructors

/-! Shape inference compares syntax. Give definitionally equal immutable
natural constexprs one spelling in the temporary elaboration input. The
original quotation and the invocation's parameter metadata stay unchanged. -/
namespace VeriTile.Triton.DSL.EquivalentCallConstants
open Lean Elab Term Meta

private partial def values (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]!::(← values value.getAppArgs[2]!)
  throwError "constexpr names must be static"

private partial def checkWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  let leaf:=raw.getKind.getString!
  if leaf.startsWith "tritonStmt" then
    for operator in ["=",":=","+=","-=","*=","/=","//=","%="] do
      if let some i:=args.findIdx? (fun arg=>arg.isAtom && arg.getAtomVal.trimAscii.toString==operator) then
        if (args.take i).any (fun arg=>arg.isIdent && names.contains arg.getId.eraseMacroScopes) then
          throwError "equivalent shape constexprs must remain immutable"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwError "a loop cannot shadow a shape constexpr"
  for child in raw.getArgs do checkWrites names child

private partial def replace (aliases : List (Name × Name)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    if let some name:=aliases.lookup raw.getId.eraseMacroScopes then mkIdentFrom raw name else raw
  else raw.setArgs (raw.getArgs.map (replace aliases))

def prepare (call : Syntax) (body : Syntax) : TermElabM Syntax := do
  let profile ← elabTermEnsuringType call[1] (mkConst ``CallSignature)
  let mut canonical:List (Name × Expr):=[]
  let mut aliases:List (Name × Name):=[]
  for item in ← values (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf item | throwError "constexpr names must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    unless ← isDefEq (← inferType value) (mkConst ``Nat) do continue
    let mut target:=none
    for (previous,known) in canonical do
      if ← isDefEq value known then target:=some previous; break
    if let some selected:=target then aliases:=aliases++[(name.toName,selected)]
    else canonical:=canonical++[(name.toName,value)]
  if aliases.isEmpty then return body
  checkWrites (canonical.map Prod.fst++aliases.map Prod.fst) body
  return replace aliases body
end VeriTile.Triton.DSL.EquivalentCallConstants
