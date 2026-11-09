import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.WelfordSource

/-! Call-side ordered sum trees for copied single-row source reductions.
The original `tl.sum(x, 1)[:, None]` remains in the source body. Its caller
selects the exact FP32 addition tree; the existing typed ordered-sum constructor
retains its [1,1] result. Every selected destination must occur exactly once.
Only closed natural constexpr values are specialized; runtime values stay in
source. A wrong axis, missing binding, mutation or shadowing fails closed. -/
namespace VeriTile.Triton.DSL.OrderedSumCalls
open Lean Elab Term Meta

syntax (name := withTrees) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " ordered_sum_trees " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "ordered-sum profile requires statically known constexpr names"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def constants (values : List (Name × TSyntax `num)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,value) := values.find? (fun p => p.1==id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| $value:num)).raw
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if ((args.take i).flatMap ids).any (fun name => values.any (fun p => p.1==name)) then
        Macro.throwError "specialized natural constexpr arguments must remain immutable"
  if let first::index::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids index).any (fun name => values.any (fun p => p.1==name)) then
      Macro.throwError "a loop cannot shadow a specialized natural constexpr"
  return raw.setArgs (← raw.getArgs.mapM (constants values))

private partial def lower (trees : List (Name × Ident)) (raw : Syntax) : MacroM (Syntax × List Name) := do
  if raw.getKind==choiceKind then return ← lower trees raw[0]!
  if let `(tritonStmt| $out:ident = $value:tritonExpr) := raw then
    if let some (name,tree) := trees.find? (fun p => p.1==out.getId.eraseMacroScopes) then
      let input ← match value with
        | `(tritonExpr| tl.sum($input:tritonExpr, 1)[:, None]) => pure input
        | `(tritonExpr| tl.sum($input:tritonExpr, axis=1)[:, None]) => pure input
        | _ => Macro.throwErrorAt value "ordered sum requires the complete tl.sum(input, 1)[:, None] assignment"
      -- The ordered constructor produces [1,1]. The source sum produces [1]
      -- before its original [:,None] view. Retain that view explicitly, so
      -- downstream shapes agree with the source's singleton-axis syntax.
      let value ← `(tritonExpr| tl.reshape(
        tl.sum($input:tritonExpr, axis=1, tree=$($tree:ident), keep_dims=true), [1])[:, None])
      return ((← `(tritonStmt| $out:ident = $value:tritonExpr)).raw,[name])
  if raw.getArgs[0]?.any (fun x => x.isAtom && x.getAtomVal=="tl.sum(") then
    Macro.throwErrorAt raw "every source sum requires an exact ordered-tree binding"
  let mut args := #[]
  let mut used := []
  for child in raw.getArgs do
    let (child,names) ← lower trees child
    args:=args.push child
    used:=used++names
  return (raw.setArgs args,used)

@[term_elab withTrees] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut values : List (Name × TSyntax `num) := []
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr names must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then
      let value ← whnf value
      if let .lit (.natVal n) := value then
        values:=values++[(name.toName,⟨Syntax.mkNumLit (toString n)⟩)]
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok value => pure value | .error message => throwError "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value => pure value | .error message => throwError "{message}"
  let parsed ← liftMacroM (constants values parsed)
  let mut bindings : List (Name × Expr) := []
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $out:ident := $value:term) := binding
      | throwError "expected a named ordered sum tree"
    let name := out.getId.eraseMacroScopes
    if bindings.any (fun p => p.1==name) then throwError "duplicate ordered sum destination"
    let value ← elabTerm value none
    let type ← whnf (← inferType value)
    unless type.isAppOfArity ``WelfordSource.Tree 1 do
      throwErrorAt binding "an ordered sum binding requires a WelfordSource.Tree"
    bindings:=bindings++[(name,value)]
  if bindings.isEmpty then throwError "at least one ordered sum tree is required"
  let occupied := ids parsed ++ ids stx
  let rec freeze (remaining : List (Name × Expr)) (trees : List (Name × Ident)) : TermElabM Expr := do
    match remaining with
    | (name,value)::rest =>
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName:=Name.mkSimple ("veritile_sum_tree_"++(fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless occupied.contains localName do break
      withLetDecl localName (← inferType value) value fun bound => do
        let result ← freeze rest (trees++[(name,mkIdent localName)])
        mkLetFVars #[bound] result
    | [] =>
      let (body,used) ← liftMacroM (lower trees parsed)
      unless trees.all (fun p => (used.filter (·==p.1)).length==1) do
        throwError "each ordered sum destination must occur exactly once"
      let `(triton { $stmts:tritonStmt* }) := body | throwError "expected a complete source body"
      let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
      let base:=mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
      PythonFullConstructors.elaborateInvocation base expected
  freeze bindings []

end VeriTile.Triton.DSL.OrderedSumCalls
