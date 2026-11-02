import VeriTile.Triton.DSL.ConditionalPureJitFunctions

/-! Rank-one block pointers with signed strides and allocation-relative bases.
The source quotation and unary helper remain intact. Construction freezes the
logical coordinates and boundary mask before forming each complete signed
address; conversion to a cell index happens only after that sum. -/
namespace VeriTile.Triton.DSL.SignedRankOneBlockPointers
open Lean Elab Term Meta

syntax (name := rankOneBlockPointer) "tl.make_block_ptr(" tritonExpr ", " "(" tritonExpr,*,? ")" ", "
  "(" tritonExpr,*,? ")" ", " "(" tritonExpr,*,? ")" ", " "(" tritonExpr,*,? ")" ", "
  ident "=" "(" tritonExpr,*,? ")" ")" : tritonExpr
syntax (name := rankOneBoundary) "boundary_check=(" ident "," ")" : tritonMemKwarg
syntax (name := withSignedBlockCall) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " signed_pointer_offsets " "[" tritonCallBinding,* "]"
  " pure_functions " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def identifiers (s : Syntax) : List Name :=
  if s.isIdent then [s.getId.eraseMacroScopes] else s.getArgs.toList.flatMap identifiers
private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "block-pointer caller metadata must have static names"
private def textValue (e : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf e | throwError "parameter name must be literal"
  return text
private partial def immutable (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let reject (name : Ident) : TermElabM Unit := do
    if names.contains name.getId.eraseMacroScopes then
      throwError "queried block-pointer parameters must remain immutable"
  match raw with
  | `(tritonStmt| $name:ident = $_:tritonExpr)
  | `(tritonStmt| $name:ident += $_:tritonExpr)
  | `(tritonStmt| $name:ident -= $_:tritonExpr)
  | `(tritonStmt| $name:ident *= $_:tritonExpr)
  | `(tritonStmt| $name:ident /= $_:tritonExpr)
  | `(tritonStmt| $name:ident := $_:tritonExpr)
  | `(tritonStmt| $name:ident : tl.constexpr = $_:tritonExpr) => reject name
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
    for name in #[a,b]++more do reject name
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = ($_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*)) =>
    for name in #[a,b]++more do reject name
  | _ => pure ()
  let args := raw.getArgs.toList
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        (identifiers b).any names.contains then throwError "loop shadows a block-pointer parameter"
  for child in raw.getArgs do immutable names child

private structure Descriptor where
  base : Ident
  mask : Ident
private abbrev Blocks := List (Name × Descriptor)
private abbrev LowerM := StateRefT Nat TermElabM
private def fresh (used : List Name) : LowerM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_block_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_block_"++toString n))

private def axisZero (constants : List Name) (value : TSyntax `tritonExpr) : TermElabM Unit := do
  match value with
  | `(tritonExpr| 0) => pure ()
  | `(tritonExpr| $name:ident) =>
    unless constants.contains name.getId.eraseMacroScopes do
      throwError "rank-one block order must be an immutable caller constexpr zero"
    let e ← elabTermEnsuringType (⟨name.raw⟩ : TSyntax `term) (mkConst ``Nat)
    unless ← isDefEq e (mkNatLit 0) do
      throwError "rank-one block order must be an immutable caller constexpr zero"
  | _ => throwError "rank-one block order must be an immutable caller constexpr zero"

private def checkBoundary (constants : List Name) (kwarg : TSyntax `tritonMemKwarg) : TermElabM Unit := do
  if kwarg.raw.getKind == ``rankOneBoundary then
    let name : Ident := ⟨kwarg.raw[1]⟩
    return ← axisZero constants (← `(tritonExpr| $name:ident))
  if let `(tritonMemKwarg| boundary_check=($items:num,*)) := kwarg then
    unless items.getElems.size == 1 && items.getElems[0]!.getNat == 0 do
      throwError "rank-one boundary check must select axis zero"
    return
  throwError "rank-one boundary check must select axis zero"

private partial def expression (constants pointers : List Name) (blocks : Blocks) (raw : Syntax) : TermElabM Syntax := do
  if let `(tritonExpr| $callee:ident($target:tritonExpr)) := raw then
    let name := callee.getId.eraseMacroScopes.toString
    if name.endsWith ".to" then
      if let `(tritonExpr| $dtype:ident) := target then
        let dtypeName := dtype.getId.eraseMacroScopes.toString
        if dtypeName.endsWith ".type.element_ty" || dtypeName.endsWith ".dtype.element_ty" then
          let receiver : TSyntax `tritonExpr := ⟨mkIdent (name.dropEnd 3).toString.toName⟩
          let receiver : TSyntax `tritonExpr := ⟨← expression constants pointers blocks receiver.raw⟩
          let dtype : TSyntax `term := ⟨← expression constants pointers blocks dtype.raw⟩
          return (← `(tritonExpr| ($receiver:tritonExpr).to($dtype:term))).raw
  if let `(tritonExpr| tl.load($pointer:ident, $kwarg:tritonMemKwarg)) := raw then
    if let some desc := blocks.lookup pointer.getId.eraseMacroScopes then
      checkBoundary constants kwarg
      let mask := desc.mask
      return (← `(tritonExpr| tl.load($pointer:ident, mask=$mask:ident))).raw
  if raw.isIdent then
    let name := raw.getId.eraseMacroScopes.toString
    for suffix in [".type.element_ty",".dtype.element_ty"] do
      if name.endsWith suffix then
        let root := (name.dropEnd suffix.length).toString.toName
        if let some desc := blocks.lookup root then
          return (mkIdent (desc.base.getId.eraseMacroScopes.toString++suffix).toName).raw
        unless pointers.contains root do
          throwError "pointer dtype cast requires known caller storage metadata"
  if raw.isIdent && (blocks.lookup raw.getId.eraseMacroScopes).isSome then
    throwError "rank-one block pointers require supported boundary-checked load or store"
  if raw.isIdent && pointers.contains raw.getId.eraseMacroScopes then
    throwError "signed caller pointers require supported rank-one block construction"
  return raw.setArgs (← raw.getArgs.mapM (expression constants pointers blocks))

private abbrev Functions := List (Name × Name × Syntax)
private partial def substitute (parameter : Name) (argument : Ident) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← substitute parameter argument raw[0]!
  if raw.isIdent && raw.getId.eraseMacroScopes == parameter then return argument.raw
  let raw := raw.setArgs (← raw.getArgs.mapM (substitute parameter argument))
  let isInput (value : TSyntax `tritonExpr) := match value with
    | `(tritonExpr| $name:ident) => name.getId.eraseMacroScopes == argument.getId.eraseMacroScopes
    | _ => Bool.false
  let zero : MacroM (TSyntax `tritonExpr) := do
    let dtype : TSyntax `term := ⟨mkIdent (argument.getId.eraseMacroScopes ++ `dtype)⟩
    `(tritonExpr| (0.0).to($dtype:term))
  -- Promote only a zero whose neighbouring tensor fixes its type. A bare
  -- integer return keeps its integer type at the call-assignment boundary.
  if let `(tritonExpr| $value:tritonExpr > 0) := raw then
    if isInput value then
      let promoted ← zero
      return (← `(tritonExpr| $value:tritonExpr > $promoted:tritonExpr)).raw
  if let `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, 0)) := raw then
    if isInput yes then
      let promoted ← zero
      return (← `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, $promoted:tritonExpr))).raw
  return raw
private def rejectFunctions (functions : Functions) (raw : Syntax) : TermElabM Unit := do
  if (identifiers raw).any (fun name => functions.any (fun f => f.1 == name)) then
    throwError "pure JIT calls require a complete plain assignment at the call site"

private partial def writes (raw : Syntax) : List Name :=
  if let `(tritonStmt| $id:ident = $_:tritonExpr) := raw then [id.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap writes

private partial def lower (functions : Functions) (used pointers constants : List Name) (offsets : List (Name × Ident))
    (blocks : Blocks) (statements : Array (TSyntax `tritonStmt)) : LowerM (Array (TSyntax `tritonStmt) × Blocks) := do
  let mut result := #[]
  let mut blocks := blocks
  for statement in statements do
    if let `(tritonStmt| $target:ident = $callee:ident($argument:tritonExpr)) := statement then
      if let some (_,parameter,body) := functions.find? (·.1 == callee.getId.eraseMacroScopes) then
        let `(tritonExpr| $argument:ident) := argument
          | throwError "pure JIT call arguments must be register names"
        if (blocks.lookup argument.getId.eraseMacroScopes).isSome || pointers.contains argument.getId.eraseMacroScopes then
          throwError "pure JIT arguments cannot be block or caller pointers"
        let input ← fresh used
        let output ← fresh used
        let value : TSyntax `tritonExpr := ⟨← liftMacroM (substitute parameter input body)⟩
        let value ← match value with
          | `(tritonExpr| $literal:num) => `(tritonExpr| tl.full((), $literal:num, tl.int64))
          | _ => pure value
        result := result ++ #[← `(tritonStmt| $input:ident = $argument:ident),
          ← `(tritonStmt| $output:ident = $value:tritonExpr),
          ← `(tritonStmt| $target:ident = $output:ident)]
        blocks := blocks.filter (·.1 != target.getId.eraseMacroScopes)
        continue
    match statement with
    | `(tritonStmt| $name:ident = tl.make_block_ptr($base:tritonExpr, ($parents:tritonExpr,*),
        ($strides:tritonExpr,*), ($coords:tritonExpr,*), ($widths:tritonExpr,*), $key:ident=($axes:tritonExpr,*))) =>
      rejectFunctions functions statement.raw
      unless key.getId.eraseMacroScopes == `order do throwError "expected block-pointer order keyword"
      let #[parent] := parents.getElems | throwError "signed block lowering requires rank one"
      let #[stride] := strides.getElems | throwError "signed block lowering requires rank one"
      let #[offset] := coords.getElems | throwError "signed block lowering requires rank one"
      let #[width] := widths.getElems | throwError "signed block lowering requires rank one"
      let #[orderAxis] := axes.getElems | throwError "signed block lowering requires rank one"
      let `(tritonExpr| $pointer:ident) := base
        | throwError "rank-one block base must be an immutable caller pointer"
      unless pointers.contains pointer.getId.eraseMacroScopes do
        throwError "rank-one block base is not a caller pointer"
      axisZero constants orderAxis
      let some (_,entry) := offsets.find? (·.1 == pointer.getId.eraseMacroScopes)
        | throwError "rank-one block base needs an explicit signed caller offset"
      let parent : TSyntax `tritonExpr := ⟨← expression constants pointers blocks parent.raw⟩
      let stride : TSyntax `tritonExpr := ⟨← expression constants pointers blocks stride.raw⟩
      let offset : TSyntax `tritonExpr := ⟨← expression constants pointers blocks offset.raw⟩
      let width : TSyntax `tritonExpr := ⟨← expression constants pointers blocks width.raw⟩
      let coords ← fresh used
      let mask ← fresh used
      result := result.push (← `(tritonStmt| $coords:ident = tl.cast($offset:tritonExpr, tl.int64) + tl.arange(0, $width:tritonExpr).to(tl.int64)))
      result := result.push (← `(tritonStmt| $mask:ident = ($coords:ident >= 0) & ($coords:ident < $parent:tritonExpr)))
      result := result.push (← `(tritonStmt| $name:ident = $pointer:ident + ($(($entry:ident : Int)) + $coords:ident * $stride:tritonExpr)))
      blocks := (name.getId.eraseMacroScopes,⟨pointer,mask⟩)::blocks.filter (·.1 != name.getId.eraseMacroScopes)
    | `(tritonStmt| tl.store($pointer:ident, $value:tritonExpr, $kwarg:tritonMemKwarg)) =>
      rejectFunctions functions statement.raw
      let some desc := blocks.lookup pointer.getId.eraseMacroScopes
        | throwError "boundary-checked store requires a locally constructed rank-one block pointer"
      checkBoundary constants kwarg
      let value : TSyntax `tritonExpr := ⟨← expression constants pointers blocks value.raw⟩
      let mask := desc.mask
      result := result.push (← `(tritonStmt| tl.store($pointer:ident, $value:tritonExpr, mask=$mask:ident)))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      rejectFunctions functions condition.raw
      let (yes,_) ← lower functions used pointers constants offsets blocks yes
      let (no,_) ← lower functions used pointers constants offsets blocks no
      let condition : TSyntax `tritonExpr := ⟨← expression constants pointers blocks condition.raw⟩
      result := result.push (← `(tritonStmt| if $condition { $yes* } else { $no* }))
      blocks := blocks.filter (fun b => !(writes statement.raw).contains b.1)
    | `(tritonStmt| for $counter:ident in range(0, $stop:tritonExpr) { $body:tritonStmt* }) =>
      rejectFunctions functions stop.raw
      let (body,_) ← lower functions used pointers constants offsets blocks body
      result := result.push (← `(tritonStmt| for $counter:ident in range(0, $stop:tritonExpr) { $body* }))
      blocks := blocks.filter (fun b => !(writes statement.raw).contains b.1)
    | `(tritonStmt| for $counter:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
      rejectFunctions functions start.raw
      rejectFunctions functions stop.raw
      let (body,_) ← lower functions used pointers constants offsets blocks body
      result := result.push (← `(tritonStmt| for $counter:ident in range($start:tritonExpr, $stop:tritonExpr) { $body* }))
      blocks := blocks.filter (fun b => !(writes statement.raw).contains b.1)
    | _ =>
      rejectFunctions functions statement.raw
      result := result.push ⟨← expression constants pointers blocks statement.raw⟩
      blocks := blocks.filter (fun b => !(writes statement.raw).contains b.1)
  return (result,blocks)

@[term_elab withSignedBlockCall] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut pointers := []
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      pointers := pointers++[(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  let constants ← (← listValues (← mkAppM ``CallSignature.constexpr #[profile])).mapM fun e =>
    return (← textValue e).toName
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw := prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed | .error message => throwError "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete block-pointer caller"
  immutable (pointers++constants) parsed
  let mut offsets := []
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding | throwError "expected a signed pointer binding"
    let name := name.getId.eraseMacroScopes
    unless pointers.contains name && !offsets.any (·.1 == name) do
      throwError "signed offsets must name distinct caller pointers"
    offsets := offsets++[(name,← elabTermEnsuringType value (mkConst ``Int))]
  let mut declared := pointers++constants
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    declared := declared++[(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  let mut functions : Functions := []
  for binding in stx[12].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding
      | throwError "expected a copied unary helper binding"
    let key := name.getId.eraseMacroScopes
    unless key.getPrefix.isAnonymous && !["tl","triton","libdevice"].contains key.toString &&
        !declared.contains key && !functions.any (·.1 == key) do
      throwError "duplicate or captured pure JIT helper name"
    let value ← elabTermEnsuringType value (mkConst ``PureJitFunction)
    let parameter ← textValue (← mkAppM ``PureJitFunction.parameter #[value])
    let source ← textValue (← mkAppM ``PureJitFunction.source #[value])
    let checked ← match Parser.runParserCategory (← getEnv) `term
        ("triton_function ("++parameter++") {\n"++source++"\n}") with
      | .ok checked => pure checked | .error message => throwError "{message}"
    unless checked.getKind == ``PureJitFunctions.pureFunction do
      throwError "expected a complete copied unary helper definition"
    let checked ← withoutErrToSorry <| elabTermEnsuringType checked (mkConst ``PureJitFunction)
    let parameter ← textValue (← mkAppM ``PureJitFunction.parameter #[checked])
    let source ← textValue (← mkAppM ``PureJitFunction.source #[checked])
    let body ← match Parser.runParserCategory (← getEnv) `tritonExpr
        (source.trimAscii.toString.drop 7).toString with
      | .ok body => pure body | .error message => throwError "{message}"
    functions := functions++[(key,parameter.toName,body)]
  immutable (functions.map (·.1)) parsed
  let used := identifiers parsed ++ identifiers stx ++ declared
  let rec freeze (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::rest =>
      let mut fresh := Name.anonymous
      repeat
        let next ← mkFreshId
        fresh := Name.mkSimple ("veritile_block_base_"++(next.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains fresh do break
      withLetDecl fresh (mkConst ``Int) value fun bound => do
        let result ← freeze rest (values++[(name,mkIdent fresh)])
        mkLetFVars #[bound] result
    | [] =>
      let ((statements,_),_) ← (lower functions used pointers constants values [] statements).run 0
      let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
        (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
      let call := mkNode ``CallSignature.withCall
        #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[14],stx[15],raw]
      CallSignature.elaborateInvocation call expected
  freeze offsets []
/-- Resolve each pointer-element cast using the exact dtype declaration emitted
by CallSignature, including its shared floating-storage alias. This preserves
explicit casts without changing the storage or compute carrier. -/
private def pointerTarget? (formats : List (Name × TSyntax `tritonDType))
    (name : Name) : Option (TSyntax `tritonDType) := do
  let text := name.eraseMacroScopes.toString
  let suffix ← [".type.element_ty",".dtype.element_ty"].find? (fun suffix => text.endsWith suffix)
  formats.lookup (text.dropEnd suffix.length).toString.toName

private partial def typedCasts (formats : List (Name × TSyntax `tritonDType))
    (raw : Syntax) : MacroM (Syntax × Bool) := do
  if let `(tritonExpr| $method:ident($target:tritonExpr)) := raw then
    if let .str receiver "to" := method.getId.eraseMacroScopes then
      if let `(tritonExpr| $name:ident) := target then
        if let some dtype := pointerTarget? formats name.getId then
          let value := mkIdent receiver
          return ((← `(tritonExpr| ($value:ident).to($dtype:tritonDType))).raw,Bool.true)
  let attributeCasts := [``tritonMethodCastElementTy,``tritonMethodCastDTypeIdent,
    ``tritonMethodCastElementTyIdent,``tritonIdentMethodCastDTypeIdent,
    ``tritonIdentMethodCastDTypeIdentSpaced,``tritonIdentMethodCastElementTyIdent,
    ``tritonIdentMethodCastElementTyIdentSpaced]
  if attributeCasts.contains raw.getKind then
    if let some target := methodCastTargetName? ⟨raw⟩ then
      if let some dtype := pointerTarget? formats target then
        let (value,_) ← typedCasts formats raw[0]
        let value : TSyntax `tritonExpr := ⟨value⟩
        return ((← `(tritonExpr| ($value:tritonExpr).to($dtype:tritonDType))).raw,Bool.true)
  let mut changed := Bool.false
  let mut children := #[]
  for child in raw.getArgs do
    let (child,next) ← typedCasts formats child
    children := children.push child
    changed := changed || next
  return (raw.setArgs children,changed)

@[term_elab tritonBlock] def elaboratePointerCasts : TermElab := fun stx expected => do
  let `(triton { $statements:tritonStmt* }) := stx | throwUnsupportedSyntax
  unless statements.any (fun statement => match statement with
      | `(tritonStmt| tl.call_signature) => Bool.true | _ => Bool.false) do throwUnsupportedSyntax
  let mut formats := []
  for statement in statements do
    if let `(tritonStmt| tl.region $name:ident = $dtype:tritonDType $[, $more:ident = $dtypes:tritonDType]*) := statement then
      formats := formats++[(name.getId.eraseMacroScopes,dtype)]++
        (more.zip dtypes).toList.map (fun (n,d) => (n.getId.eraseMacroScopes,d))
  let (next,changed) ← liftMacroM (typedCasts formats stx)
  unless changed do throwUnsupportedSyntax
  elabTerm next expected

end VeriTile.Triton.DSL.SignedRankOneBlockPointers
