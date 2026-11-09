import VeriTile.Triton.DSL.StatementJitFunctions
import VeriTile.Triton.DSL.JitIntegerPromotion

/-! Triton also specializes unannotated helper parameters when every actual
argument is a caller constexpr. Propagate only immutable named constexprs to
read-only helper parameters. The complete helper/caller quotations stay intact;
runtime values and parameters assigned by the helper keep ordinary captures.
This optional adapter leaves the base helper frontend unchanged. -/
namespace VeriTile.Triton.DSL.StatementJitCallConstants
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "JIT specialization metadata must be static"
private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "JIT specialization names and source must be static"
  return text
private def names (value : Expr) : TermElabM (List Name) := do
  (← listValues value).mapM fun value=>return (← stringValue value).toName

private partial def firstChoice (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then firstChoice raw[0]! else raw

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let text:=PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body=>pure body | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }) := parsed | throwError "complete JIT body required"
  return body.map fun stmt=>⟨firstChoice stmt.raw⟩

private def calls (name : Name) (parameters : List Name) (body : Array (TSyntax `tritonStmt)) :
    TermElabM (List (List (Name × TSyntax `tritonExpr))) := do
  let mut found:=[]
  for stmt in body do
    let selected ← match stmt with
      | `(tritonStmt| $_:ident = $callee:ident($args:statementJitArgument,*)) =>
        pure (some (callee,args.getElems))
      | `(tritonStmt| $_:ident, $_:ident $[, $_:ident]* = $callee:ident($args:statementJitArgument,*)) =>
        pure (some (callee,args.getElems))
      | _=>pure none
    let some (callee,args):=selected | continue
    unless callee.getId.eraseMacroScopes==name do continue
    let mut mapped:List (Name × TSyntax `tritonExpr):=[]
    let mut positional:=0
    for arg in args do
      match arg with
      | `(statementJitArgument| $key:ident = $value:tritonExpr) =>
        mapped:=mapped++[(key.getId.eraseMacroScopes,value)]
      | `(statementJitArgument| $value:tritonExpr) =>
        if let some parameter:=parameters[positional]? then mapped:=mapped++[(parameter,value)]
        positional:=positional+1
      | _=>pure ()
    found:=found++[mapped]
  return found

private partial def substituteIntegers (values : List (Name × Nat)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,n):=values.find? (·.1==id.getId.eraseMacroScopes) then
      let numeral:TSyntax `num:=⟨Syntax.mkNumLit (toString n)⟩
      return (← `(tritonExpr| $numeral:num)).raw
  return raw.setArgs (← raw.getArgs.mapM (substituteIntegers values))

private partial def substituteAliases (values : List (Name × Syntax)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident):=raw then
    if let some (_,value):=values.find? (·.1==id.getId.eraseMacroScopes) then
      let value:TSyntax `tritonExpr:=⟨value⟩
      return (← `(tritonExpr| ($value:tritonExpr))).raw
  return raw.setArgs (← raw.getArgs.mapM (substituteAliases values))

private partial def naturalShape (constants : List Name) (value : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match value with
  | `(tritonExpr| $n:num)=>return ⟨n.raw⟩
  | `(tritonExpr| $id:ident)=>
    unless constants.contains id.getId.eraseMacroScopes do Macro.throwError "shape arithmetic requires immutable constexprs"
    return ⟨id.raw⟩
  | `(tritonExpr| ($inner:tritonExpr))=>naturalShape constants inner
  | `(tritonExpr| $left:tritonExpr * $right:tritonExpr)=>
    let left ← naturalShape constants left;let right ← naturalShape constants right
    `($left * $right)
  | `(tritonExpr| $left:tritonExpr + $right:tritonExpr)=>
    let left ← naturalShape constants left;let right ← naturalShape constants right
    `($left + $right)
  | _=>Macro.throwError "unsupported natural constexpr shape expression"

private partial def shapeRanges (constants : List Name) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| tl.arange(0, $stop:tritonExpr)):=raw then
    match stop with
    | `(tritonExpr| $_:ident) | `(tritonExpr| $_:num)=>return raw
    | _=>
      let value ← naturalShape constants stop
      return (← `(tritonExpr| tl.arange(0, $($value:term)))).raw
  return raw.setArgs (← raw.getArgs.mapM (shapeRanges constants))

private def inlineAliases (constants : List Name) (body : Array (TSyntax `tritonStmt)) :
    TermElabM (Array (TSyntax `tritonStmt)) := do
  -- Validate the original scope/writes before propagating any alias.
  let _ ← liftMacroM (RawBodyConstants.prepare constants [] body.toList)
  let mut values:List (Name × Syntax):=[]
  let mut out:=#[]
  for stmt in body do
    let normalized:TSyntax `tritonStmt:=⟨← liftMacroM (substituteAliases values stmt.raw)⟩
    out:=out.push ⟨← liftMacroM (shapeRanges constants normalized.raw)⟩
    if let `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr):=normalized then
      values:=values++[(name.getId.eraseMacroScopes,value.raw)]
  return out

/-- Closed Python integers retain weak-literal promotion. Shift counts use
checked constexpr aliases because the signed-shift IR expects a Nat count. -/
private partial def foldIntegers (values : List (Name × Nat)) (used : List Name) (raw : Syntax) :
    StateT (Array (Nat × Ident)) MacroM Syntax := do
  if let some (value,dtype):=methodBitcast? ⟨raw⟩ then
    return (← `(tritonExpr| tl.bitcast($value:tritonExpr, $dtype:tritonDType))).raw
  if let `(tritonExpr| $callee:ident($args:statementJitArgument,*)):=raw then
    let spelling:=callee.getId.eraseMacroScopes.toString
    if spelling.endsWith ".to" && args.getElems.size==2 then
      let `(statementJitArgument| $target:tritonExpr):=args.getElems[0]! | Macro.throwError "bitcast dtype required"
      if let `(statementJitArgument| bitcast = true):=args.getElems[1]! then
        let dtype ← match (PythonFullConstructors.render target.raw).trimAscii.toString with
          | "tl.float32"=>`(tritonDType| tl.float32)
          | "tl.int32"=>`(tritonDType| tl.int32)
          | "tl.uint32"=>`(tritonDType| tl.uint32)
          | _=>Macro.throwError "unsupported statement-helper bitcast target"
        let receiver:=mkIdent (spelling.dropEnd 3).toString.toName
        return (← `(tritonExpr| tl.bitcast($receiver:ident, $dtype:tritonDType))).raw
  let shifted:=match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| $left:tritonExpr << $right:tritonExpr)=>some (left,right,Bool.true)
    | `(tritonExpr| $left:tritonExpr >> $right:tritonExpr)=>some (left,right,Bool.false)
    | _=>none
  if let some (left,right,direction):=shifted then
    let replaced ← liftM <| substituteIntegers values right.raw
    if let .ok (some value):=PythonConstants.evaluate ⟨replaced⟩ then
      if !value.floating && value.ideal.den==1 && value.ideal.num>=0 then
        let aliases ← get
        let id ← if let some (_,id):=aliases.find? (·.1==value.ideal.num.natAbs) then pure id else do
          let mut index:=aliases.size
          while used.contains (Name.mkSimple ("__veritile_folded_integer_"++toString index)) ||
              aliases.any (fun p=>p.2.getId==Name.mkSimple ("__veritile_folded_integer_"++toString index)) do
            index:=index+1
          let id:=mkIdent (Name.mkSimple ("__veritile_folded_integer_"++toString index))
          set (aliases.push (value.ideal.num.natAbs,id))
          pure id
        let left:TSyntax `tritonExpr:=⟨← foldIntegers values used left.raw⟩
        if direction then return (← `(tritonExpr| $left:tritonExpr << $id:ident)).raw
        else return (← `(tritonExpr| $left:tritonExpr >> $id:ident)).raw
  if let `(tritonExpr| $id:ident):=raw then
    if let some (_,value):=values.find? (·.1==id.getId.eraseMacroScopes) then
      let numeral:TSyntax `num:=⟨Syntax.mkNumLit (toString value)⟩
      return (← `(tritonExpr| $numeral:num)).raw
  if let some (operand,dtype):=methodCast? ⟨raw⟩ then
    let operand:TSyntax `tritonExpr:=⟨← foldIntegers values used operand.raw⟩
    return (← `(tritonExpr| tl.cast($operand:tritonExpr, $dtype:tritonDType))).raw
  return raw.setArgs (← raw.getArgs.mapM (foldIntegers values used))

@[term_elab StatementJitFunctions.withFunctions]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let constants ← names (← mkAppM ``CallSignature.constexpr #[profile])
  let mut bytePointers:List Name:=[]
  for pair in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
    let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
    if dtype.isConstOf ``VeriTile.Triton.SourceDType.u8 then
      bytePointers:=bytePointers++[(← stringValue (← mkAppM ``Prod.fst #[pair])).toName]
  let mut integers:List (Name × Nat):=[]
  for name in constants do
    let value ← elabTerm (mkIdent name) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then
      if let .lit (.natVal n) ← whnf value then integers:=integers++[(name,n)]
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok text=>pure text | .error message=>throwError "{message}"
  let body ← inlineAliases constants (← parse source)
  let mut bindings:List (Ident × Expr):=[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding | throwError "named JIT helper required"
    let helper ← elabTermEnsuringType value (mkConst ``StatementJitFunction)
    let parameters ← names (← mkAppM ``StatementJitFunction.parameters #[helper])
    let declared ← names (← mkAppM ``StatementJitFunction.constexpr #[helper])
    let source ← stringValue (← mkAppM ``StatementJitFunction.source #[helper])
    let helperBody ← parse source
    let written:=helperBody.toList.filterMap fun stmt=>match stmt with
      | `(tritonStmt| $id:ident = $_:tritonExpr)
      | `(tritonStmt| $id:ident : tl.constexpr = $_:tritonExpr) => some id.getId.eraseMacroScopes
      | _=>none
    let occurrences ← calls id.getId.eraseMacroScopes parameters body
    let inferred:=parameters.filter fun name=>
      !written.contains name && !occurrences.isEmpty && occurrences.all (fun args=>
        (args.find? (·.1==name)).any fun (_,value)=>match value with
          | `(tritonExpr| $actual:ident)=>constants.contains actual.getId.eraseMacroScopes
          | _=>Bool.false)
    let mut known:List (Name × Nat):=[]
    for name in (declared++inferred).eraseDups do
      if written.contains name then continue
      let values:=occurrences.filterMap fun args=>do
        let (_,value) ← args.find? (·.1==name)
        let `(tritonExpr| $actual:ident):=value | none
        (integers.find? (·.1==actual.getId.eraseMacroScopes)).map Prod.snd
      if let n::rest:=values then
        if values.length==occurrences.length && rest.all (·==n) then known:=known++[(name,n)]
    let normalizer : StateT (Array (Nat × Ident)) MacroM (Array Syntax) :=
      helperBody.mapM fun stmt=>foldIntegers known (helperBody.toList.flatMap (ids ∘ TSyntax.raw)++parameters) stmt.raw
    let (normalized,aliases) ← liftMacroM (normalizer.run #[])
    let mut before:Array Syntax:=#[]
    for (value,id) in aliases do
      let numeral:TSyntax `num:=⟨Syntax.mkNumLit (toString value)⟩
      before:=before.push (← `(tritonStmt| $id:ident : tl.constexpr = $numeral:num)).raw
    let byteCalls:=JitIntegerPromotion.byteParameters bytePointers id.getId.eraseMacroScopes parameters body
    let byteInputs:=parameters.filter fun parameter=>!byteCalls.isEmpty && byteCalls.all (·.contains parameter)
    let promoted ← liftMacroM (JitIntegerPromotion.normalize byteInputs (before++normalized))
    let source:=String.intercalate "\n" (promoted.toList.map PythonFullConstructors.render)
    let constantNames:List Expr:=(declared++inferred).eraseDups.map (mkStrLit ∘ Name.toString)
    let updated:=mkApp3 (mkConst ``StatementJitFunction.mk)
      (← mkAppM ``StatementJitFunction.parameters #[helper])
      (← mkListLit (mkConst ``String) constantNames) (mkStrLit source)
    bindings:=bindings++[(id,updated)]
  let rec freeze (todo : List (Ident × Expr)) (out : Array (TSyntax `tritonCallBinding)) : TermElabM Expr := do
    match todo with
    | (id,value)::rest=>
      let used:=ids stx ++ (← getLCtx).foldl (fun names decl=>decl.userName.eraseMacroScopes::names) []
      let mut index:=0
      while used.contains (Name.mkSimple ("__veritile_specialized_jit_"++toString index)) do
        index:=index+1
      let name:=Name.mkSimple ("__veritile_specialized_jit_"++toString index)
      withLetDecl name (mkConst ``StatementJitFunction) value fun captured=>do
        let syntaxValue:=mkIdent name
        let binding ← `(tritonCallBinding| $id:ident := $syntaxValue:ident)
        let result ← freeze rest (out.push binding)
        mkLetFVars #[captured] result
    | []=>
      let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
        (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
      let next:=stx.setArg 8 (Syntax.node .none nullKind (out.toList.intersperse (Syntax.atom .none ",")).toArray)
      let next:=next.setArgs (next.getArgs.set! (next.getArgs.size-1) raw)
      StatementJitFunctions.elaborateInvocation next expected
  freeze bindings #[]

end VeriTile.Triton.DSL.StatementJitCallConstants
