import VeriTile.Triton.DSL.PythonFullConstructors

/-! Compose natural call dimensions inside block-pointer shape/stride tuples.
Only caller-supplied immutable metadata can enter these compile-time shape
expressions. Offsets and executable arithmetic stay in the kernel. Literal
shift counts keep their natural dtype. Static assertions must be proved from
constexprs before they can be erased as compile-time-only source statements. -/
namespace VeriTile.Triton.DSL.PythonShapeExpressions
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "shape call profile must have literal constexpr names"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let reject (id : Ident) : TermElabM Unit := do
    if names.contains id.getId.eraseMacroScopes then
      throwError "call dimensions used by shape expressions must remain immutable"
  match raw with
  | `(tritonStmt| $id:ident = $_:tritonExpr)
  | `(tritonStmt| $id:ident := $_:tritonExpr)
  | `(tritonStmt| $id:ident += $_:tritonExpr)
  | `(tritonStmt| $id:ident -= $_:tritonExpr)
  | `(tritonStmt| $id:ident *= $_:tritonExpr)
  | `(tritonStmt| $id:ident /= $_:tritonExpr)
  | `(tritonStmt| $id:ident : tl.constexpr = $_:tritonExpr) => reject id
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
    reject a
    reject b
    for id in more do reject id
  | _ => pure ()
  let args:=raw.getArgs.toList
  if let first::index::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids index).any names.contains then
      throwError "a loop cannot shadow a call dimension"
  for child in raw.getArgs do rejectWrites names child

private partial def natural (names : List (Name × Ident)) (raw : Syntax) : MacroM (TSyntax `term) := do
  let e : TSyntax `tritonExpr:=⟨raw⟩
  match e with
  | `(tritonExpr| $n:num) => `(($n:num : Nat))
  | `(tritonExpr| ($e:tritonExpr)) => natural names e
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
    let a ← natural names a; let b ← natural names b
    `(($a * $b : Nat))
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
    let a ← natural names a; let b ← natural names b
    `(($a + $b : Nat))
  | _ =>
    let name ← if raw.isIdent then pure raw.getId.eraseMacroScopes else
      match e with | `(tritonExpr| $name:ident) => pure name.getId.eraseMacroScopes
                   | _ => Macro.throwError "shape expressions require natural call dimensions, literals, + or *"
    let some (_,value) := names.find? (fun p => p.1==name)
      | Macro.throwError "shape expression cannot read a runtime local or unknown metadata"
    pure ⟨value.raw⟩

private def shapeEntry (names : List (Name × Ident)) (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  let n ← natural names e
  if e.raw.isIdent then return e
  match e with
  | `(tritonExpr| $_n:num) => return e
  | `(tritonExpr| $_id:ident) => return e
  | _ => `(tritonExpr| $($n))

private partial def lower (names : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower names raw[0]!
  if let `(tritonStmt| tl.static_assert($_:tritonExpr $[, $_:str]?)) := raw then
    Macro.throwError "static shape assertions must occur at the top level of the copied body"
  if let `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base:tritonExpr,
      $shapeKw:ident=($parent:tritonExpr,*), $strideKw:ident=($strides:tritonExpr,*),
      $offsetKw:ident=($offsets:tritonExpr,*), $blockKw:ident=($block:tritonExpr,*),
      $orderKw:ident=($order:num,*))) := raw then
    unless [baseKw,shapeKw,strideKw,offsetKw,blockKw,orderKw].map (·.getId.eraseMacroScopes.toString)==
        ["base","shape","strides","offsets","block_shape","order"] do
      Macro.throwError "block-pointer keywords differ from the source constructor"
    let shape ← parent.getElems.mapM (shapeEntry names)
    let strides ← strides.getElems.mapM (shapeEntry names)
    let block ← block.getElems.mapM (shapeEntry names)
    let offsets : Array (TSyntax `tritonExpr) ← offsets.getElems.mapM fun e => return ⟨← lower names e⟩
    return (← `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base:tritonExpr,
      $shapeKw:ident=($shape,*), $strideKw:ident=($strides,*), $offsetKw:ident=($offsets,*),
      $blockKw:ident=($block,*), $orderKw:ident=($order,*)))).raw
  if let `(tritonExpr| $input:tritonExpr << $n:num) := raw then
    return (← `(tritonExpr| $input:tritonExpr << $(($n:num : Nat)))).raw
  if let `(tritonExpr| $input:tritonExpr >> $n:num) := raw then
    return (← `(tritonExpr| $input:tritonExpr >> $(($n:num : Nat)))).raw
  return raw.setArgs (← raw.getArgs.mapM (lower names))

@[term_elab CallSignature.withCall] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``VeriTile.Triton.CallSignature)
  let mut constexpr : List Name := []
  let mut arguments : List (Name × Expr) := []
  for value in ← listValues (← mkAppM ``VeriTile.Triton.CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "constexpr name must be literal"
    constexpr:=constexpr++[name.toName]
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then arguments:=arguments++[(name.toName,value)]
  for binding in stx[4].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding | throwError "named call argument required"
    let value ← elabTerm value none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then arguments:=arguments++[(name.getId.eraseMacroScopes,value)]
  let prepared ← PythonFullConstructors.prepareSource stx
  let source:=prepared.getArgs.back!.getAtomVal
  let .ok parsed := Parser.runParserCategory (← getEnv) `term ("triton "++source)
    | throwError "expected complete shape-bearing source"
  let parsed ← liftMacroM (PositionalBlockPointer.normalize parsed)
  rejectWrites (arguments.map Prod.fst) parsed
  let used:=ids parsed++ids stx
  let rec freeze (todo : List (Name × Expr)) (names : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (source,value)::tail =>
      if constexpr.contains source then
        return ← freeze tail (names++[(source,mkIdent source)])
      let fresh ← mkFreshId
      let name:=Name.mkSimple ("veritile_shape_arg_"++(fresh.toString.map fun c => if c.isAlphanum then c else '_'))
      if used.contains name then throwError "shape argument hygiene collision"
      withLetDecl name (mkConst ``Nat) value fun captured => do
        let result ← freeze tail (names++[(source,mkIdent name)])
        mkLetFVars #[captured] result
    | [] =>
      let `(triton { $stmts:tritonStmt* }) := parsed | throwError "complete source body required"
      let mut selected := #[]
      for statement in stmts do
        if let `(tritonStmt| tl.static_assert($condition:tritonExpr)) := statement then
          let `(tritonExpr| $a:tritonExpr == $b:tritonExpr) := condition
            | throwError "static shape assertion requires equality of natural constexprs"
          if (ids condition).any (fun name => !constexpr.contains name) then
            throwError "static assertion cannot depend on runtime scalar arguments"
          let a ← liftMacroM (natural (names.filter (fun p => constexpr.contains p.1)) a)
          let b ← liftMacroM (natural (names.filter (fun p => constexpr.contains p.1)) b)
          let a ← elabTermEnsuringType a (mkConst ``Nat)
          let b ← elabTermEnsuringType b (mkConst ``Nat)
          unless ← isDefEq a b do throwError "static shape assertion is not established by the caller"
          continue
        selected:=selected.push (⟨← liftMacroM (lower names statement)⟩ : TSyntax `tritonStmt)
      let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (selected.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
      let call:=prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)
      CallSignature.elaborateInvocation call expected
  freeze arguments []

end VeriTile.Triton.DSL.PythonShapeExpressions
