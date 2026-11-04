import VeriTile.Triton.DSL.ProcedureJitFunctions

/-! Python parameter rebinding is local to its invocation. Preserve the pasted
body and lower each rebound parameter through a fresh local initialized from
the incoming argument. Calls keep positional/keyword argument names and the
existing ABI, scope, typed-load and effect checks. -/
namespace VeriTile.Triton.DSL.RebindingProcedures
open Lean Elab Term Meta
syntax (name := rebindingProcedure) "triton_rebinding_procedure " term:max " (" ident,* ")"
  " constexpr_args " "[" ident,* "]" RawSourceText.body : term
syntax (name := rebindingCalls) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " rebinding_jit_procedures " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term

private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def written (raw : Syntax) : List Name :=
  let args:=raw.getArgs
  let leaf:=raw.getKind.getString!
  let isKeyword:=leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
    leaf.startsWith "tritonReduceKwarg" || leaf.startsWith "tritonKwarg" || leaf.startsWith "statementJitArgument"
  let direct:=if isKeyword then [] else match args.findIdx? (fun arg=>arg.isAtom &&
      ["=","+=","-=","*=","/="].contains arg.getAtomVal.trimAscii.toString) with
    | some index=>(args.extract 0 index).toList.flatMap ids
    | none=>[]
  direct++args.toList.flatMap written
private partial def rename (mapping : List (Name × Name)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    match mapping.find? (fun pair=>name==pair.1 || name.toString.startsWith (pair.1.toString++".")) with
    | some (old,fresh)=>mkIdent (fresh.toString++(name.toString.drop old.toString.length).toString).toName
    | none=>raw
  else
    let args:=raw.getArgs
    let leaf:=raw.getKind.getString!
    let keywordOwner:=leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
      leaf.startsWith "statementJitArgument"
    raw.setArgs (args.mapIdx fun i arg=>
      if keywordOwner && arg.isIdent && args[i+1]?.any
          (fun next=>next.isAtom && next.getAtomVal.trimAscii.toString=="=")
      then arg else rename mapping arg)
private partial def tokens (escaped : Bool) : Syntax → List String
  | .atom _ text=>[text]
  | .ident _ spelling name _=>
    let name:=name.eraseMacroScopes.toString
    [if name=="do" || name=="_" then (if escaped then "«"++name++"»" else name) else spelling.toString]
  | .node _ kind args=>if kind==choiceKind then tokens escaped args[0]! else args.toList.flatMap (tokens escaped)
  | .missing=>[]
private def render (escaped : Bool) (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens escaped raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out
/-- `Nat → Int` widening has the same nonnegative extent. Keep the
executable widening assignment, while binding only the block-pointer's static
parent extent to the immutable incoming argument. No narrowing cast qualifies. -/
private partial def parentExtents (aliases : List (Name × Name)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base:tritonExpr,
      $shapeKw:ident=($parent:tritonExpr,*), $strideKw:ident=($strides:tritonExpr,*),
      $offsetKw:ident=($offsets:tritonExpr,*), $blockKw:ident=($block:tritonExpr,*),
      $orderKw:ident=($order:num,*))) := raw then
    unless shapeKw.getId.eraseMacroScopes.toString=="shape" do Macro.throwError "parent shape keyword required"
    let parent:=parent.getElems.map (fun e=>(⟨rename aliases e.raw⟩ : TSyntax `tritonExpr))
    return (← `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base:tritonExpr,
      $shapeKw:ident=($parent,*), $strideKw:ident=($strides,*), $offsetKw:ident=($offsets,*),
      $blockKw:ident=($block,*), $orderKw:ident=($order,*)))).raw
  return raw.setArgs (← raw.getArgs.mapM (parentExtents aliases))

/-- Function and method forms of the same signed32 cast. Parenthesize the
receiver so existing signed-offset inference retains the narrowing operation. -/
private partial def signedOffsetCasts (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM signedOffsetCasts)
  if let `(tritonExpr| tl.cast($value:tritonExpr, $dtype:tritonDType)):=raw then
    match dtype with
    | `(tritonDType| tl.int8) | `(tritonDType| tl.int16) | `(tritonDType| tl.int32)=>
      return (← `(tritonExpr| ($value:tritonExpr).to($dtype:tritonDType))).raw
    | _=>pure ()
  return raw

private def laneView (lanes : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $name:ident[:, None])
  | `(tritonExpr| $name:ident[None, :])=>lanes.contains name.getId.eraseMacroScopes
  | _=>Bool.false
private def arange (raw : Syntax) : Bool :=
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.arange(0, $_:tritonExpr))=>Bool.true
  | _=>Bool.false
private def naturalStep (constants : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $name:ident)=>constants.contains name.getId.eraseMacroScopes
  | `(tritonExpr| $_:num)=>Bool.true
  | _=>Bool.false
/-- Reject a candidate lane name globally if any assignment changes its kind.
The ordinary procedure validator separately checks initialization and scope. -/
private partial def laneAssignments (constants : List Name) (raw : Syntax) : List (Name × Bool) :=
  let direct:=match raw with
    | `(tritonStmt| $name:ident = $value:tritonExpr)=>[(name.getId.eraseMacroScopes,arange value.raw)]
    | `(tritonStmt| $name:ident += $value:tritonExpr)=>[(name.getId.eraseMacroScopes,naturalStep constants value)]
    | _=>
      let args:=raw.getArgs
      if args.size==7 && args[0]!.isIdent && args[2]!.isIdent && args[3]!.getAtomVal.trimAscii.toString=="=" then
        [(args[0]!.getId.eraseMacroScopes,arange args[4]!),
         (args[2]!.getId.eraseMacroScopes,arange args[6]!)]
      else if args.size≥2 && args[0]!.getAtomVal.trimAscii.toString=="for" && args[1]!.isIdent then
        [(args[1]!.getId.eraseMacroScopes,Bool.false)]
      else
        let leaf:=raw.getKind.getString!
        let statement:=leaf.startsWith "tritonStmt" || leaf.startsWith "tuple"
        if statement then
          match args.findIdx? (fun arg=>arg.isAtom && ["=",":=","+=","-=","*=","/="].contains arg.getAtomVal.trimAscii.toString) with
          | some index=>(args.extract 0 index).toList.flatMap ids |>.map (fun name=>(name,Bool.false))
          | none=>[]
        else []
  direct++raw.getArgs.toList.flatMap (laneAssignments constants)
private partial def signedLaneValues (lanes : List Name) (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM (signedLaneValues lanes))
  if let `(tritonExpr| -$value:tritonExpr):=raw then
    if laneView lanes value then return (← `(tritonExpr| -($value:tritonExpr).to(tl.int32))).raw
  if let `(tritonExpr| $left:tritonExpr - $right:tritonExpr):=raw then
    if laneView lanes left && laneView lanes right then
      return (← `(tritonExpr| ($left:tritonExpr).to(tl.int32) - ($right:tritonExpr).to(tl.int32))).raw
  return raw

/-- In an FP32-only invocation a Python floating literal has the default
FP32 tensor type. This materializes its type, without a conversion operation.
Explicit full constructors and FP64 source contexts retain their own typing. -/
private partial def hasWeakFloat (raw : Syntax) : Bool :=
  if let `(tritonExpr| tl.full([$_:tritonExpr,*], $_:tritonExpr, dtype=$_:tritonDType)):=raw then Bool.false
  else if let `(tritonExpr| $_:scientific):=raw then Bool.true
  else raw.getArgs.any hasWeakFloat

private partial def fp32Literals (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| tl.full([$_:tritonExpr,*], $_:tritonExpr, dtype=$_:tritonDType)):=raw then return raw
  if let `(tritonExpr| $value:scientific):=raw then
    return (← `(tritonExpr| tl.full([], $value:scientific, dtype=tl.float32))).raw
  return raw.setArgs (← raw.getArgs.mapM fp32Literals)

private partial def listEntries (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "static ABI required"
  return value.getAppArgs[1]!::(← listEntries value.getAppArgs[2]!)

private def fp32Profile (profile : Expr) : TermElabM Bool := do
  for entry in ← listEntries (← mkAppM ``CallSignature.pointers #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[entry])
    unless dtype.isConst && ![``SourceDType.fp16,``SourceDType.bf16,``SourceDType.fp64,
        ``SourceDType.f8e4,``SourceDType.f8e5].any dtype.isConstOf do return Bool.false
  for entry in ← listEntries (← mkAppM ``CallSignature.floatingPointers #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[entry])
    unless dtype.isConstOf ``FloatDType.fp32 do return Bool.false
  for entry in ← listEntries (← mkAppM ``CallSignature.scalars #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[entry])
    unless dtype.isConst && !dtype.isConstOf ``ScalarType.fp64 do return Bool.false
  return Bool.true

private def lower (source : String) (parameters : List Name) (constants : List Name)
    (procedures : List Name := []) (defaultFP32 : Bool := Bool.false) : TermElabM Syntax := do
  let base ← `(with_triton_call ({} : CallSignature) call_args [] in triton {})
  let prepared ← PythonFullConstructors.prepareSource (base.raw.setArgs
    (base.raw.getArgs.set! (base.raw.getArgs.size-1) (Syntax.atom .none source)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++prepared.getArgs.back!.getAtomVal) with
    | .ok parsed=>pure (first parsed) | .error message=>throwError "{message}"
  let parsed ← liftMacroM (PositionalBlockPointer.normalize parsed)
  let parsed:=first parsed
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete rebinding source required"
  let otherFloat:=[`tl.float16,`tl.bfloat16,`tl.float64,`tl.float8e4nv,`tl.float8e5]
  let fp32Only:=defaultFP32 && !otherFloat.any (ids parsed).contains
  if body.any (hasWeakFloat ∘ TSyntax.raw) && !fp32Only then
    throwError "weak floating literals in a mixed-format rebinding procedure require explicit promotion support"
  let writes:=body.toList.flatMap (written ∘ TSyntax.raw)
  let rebound:=parameters.filter writes.contains
  if rebound.any constants.contains then throwError "constexpr parameters cannot be rebound"
  let used:=ids parsed
  -- ABI-based scope names remain stable across assignment-fusion candidates.
  let scope:=(String.intercalate "," ((parameters++constants).map Name.toString)).hash
  let mut mapping:List (Name × Name):=[]
  let mut initialBindings:Array (TSyntax `tritonStmt):=#[]
  for name in rebound do
    let mut index:=mapping.length
    let freshName (i : Nat):=("__veritile_argument_"++toString scope++"_"++toString i).toName
    while used.contains (freshName index) || mapping.any (·.2==freshName index) do index:=index+1
    let fresh:=mkIdent (freshName index)
    let incoming:=mkIdent name
    mapping:=mapping++[(name,fresh.getId)]
    initialBindings:=initialBindings.push (← `(tritonStmt| $fresh:ident = $incoming:ident))
  let mut aliases:List (Name × Name):=[]
  for statement in body do
    if let `(tritonStmt| $destination:ident = tl.cast($incoming:ident, tl.int64)):=statement then
      let name:=destination.getId.eraseMacroScopes
      if incoming.getId.eraseMacroScopes==name && (writes.filter (·==name)).length==1 then
        if let some (_,fresh):=mapping.find? (·.1==name) then aliases:=aliases++[(fresh,name)]
  let assignments:=body.toList.flatMap (laneAssignments constants ∘ TSyntax.raw)
  let lanes:=(assignments.map Prod.fst).eraseDups.filter fun name=>
    assignments.all (fun entry=>entry.1!=name || entry.2)
  let rewritten ← body.mapM fun statement=>do
    let statement:TSyntax `tritonStmt:=⟨rename mapping statement.raw⟩
    if let `(tritonStmt| $callee:ident($arguments:statementJitArgument,*)):=statement then
      if procedures.contains callee.getId.eraseMacroScopes then
        let arguments:=arguments.getElems.map fun argument=>
          (⟨rename aliases argument.raw⟩ : TSyntax `statementJitArgument)
        return ← liftMacroM `(tritonStmt| $callee:ident($arguments,*))
    let changed ← liftMacroM do signedLaneValues lanes (← signedOffsetCasts (← parentExtents aliases statement.raw))
    let changed ← if fp32Only then
        liftMacroM (fp32Literals changed) else pure changed
    return (⟨changed⟩ : TSyntax `tritonStmt)
  let lowered:=initialBindings++rewritten
  let suite:=body.any fun statement=>
    let text:=render Bool.true statement.raw
    ["for ","if ","while "].any (fun introducer=>text.startsWith introducer)
  let text:="{\n"++String.intercalate "\n" (lowered.toList.map (render suite ∘ TSyntax.raw))++"\n}"
  return Syntax.atom .none text

@[term_elab rebindingProcedure]
def elaborateProcedure : TermElab := fun stx expected=>do
  let parameters:=stx[3].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let constants:=stx[7].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let raw ← lower stx.getArgs.back!.getAtomVal parameters constants [] (← fp32Profile profile)
  let signature:TSyntax `term:=⟨stx[1]⟩
  let parameters:=stx[3].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let constants:=stx[7].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let base ← `(triton_procedure $signature:term ($parameters,*) constexpr_args [$constants,*] {})
  ProcedureJitFunctions.elaborateProcedure
    (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)) expected

@[term_elab rebindingCalls]
def elaborateCalls : TermElab := fun stx expected=>do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let source:=stx.getArgs.back!.getAtomVal
  let mut parameters:List Name:=[]
  for entry in ← listEntries (← mkAppM ``CallSignature.scalars #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry]) | throwError "static ABI names required"
    parameters:=parameters++[name.toName]
  let callees:=stx[8].getSepArgs.toList.map (fun binding=>binding[0]!.getId.eraseMacroScopes)
  let mut constants:List Name:=[]
  for entry in ← listEntries (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "static constexpr names required"
    constants:=constants++[name.toName]
  let raw ← lower source parameters constants callees (← fp32Profile profile)
  let signature:TSyntax `term:=⟨stx[1]⟩
  let arguments:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let procedures:=stx[8].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let base ← `(with_triton_call $signature:term call_args [$arguments,*] jit_procedures [$procedures,*] in triton {})
  ProcedureJitFunctions.elaborateCalls
    (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.RebindingProcedures
