import VeriTile.Triton.DSL.GuardedJitFunctions

/-! JIT helpers whose definite assignments depend on a caller's closed
integer constexpr. Store the original body and validate each specialization
at its actual call site, then delegate argument capture and hygienic expansion
to the guarded helper frontend. Runtime tests are never selected here. -/
namespace VeriTile.Triton
structure SpecializedStatementJitFunction where
  parameters : List String
  constexpr : List String
  source : String

namespace DSL.SpecializedJitFunctions
open Lean Elab Term Meta

private def text (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok source=>pure source | .error message=>throwErrorAt raw message

private partial def tokens : Syntax → List String
  | .atom _ value=>[value]
  | .ident _ spelling name _=>
    let name:=name.eraseMacroScopes.toString
    [if name=="_" || name=="do" then name else spelling.toString]
  | .node _ kind args=>if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing=>[]
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token !="." && token !="(" && token !="[" && previous !="." then out:=out++" "
    out:=out++token
    previous:=token
  return out

private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)
private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let raw ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure value | .error message=>throwError "{message}\n{source}"
  let `(triton { $body:tritonStmt* }) := raw | throwError "expected complete JIT source"
  return body.map (fun statement=>⟨first statement.raw⟩)

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def immutable (constants : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  if args.head?.any (·.isIdent) then
    if let some i:=args.findIdx? (fun token=>token.isAtom &&
        ["=",":=","+=","-=","*=","/="].contains token.getAtomVal.trimAscii.toString) then
      if ((args.take i).flatMap ids).any constants.contains then
        throwError "specialized JIT constexpr arguments must remain immutable"
  if let head::counter::_:=args then
    if head.isAtom && ["for","tl.for"].contains head.getAtomVal.trimAscii.toString &&
        (ids counter).any constants.contains then throwError "a loop cannot shadow a constexpr"
  for child in raw.getArgs do immutable constants child

private partial def integer? (values : List (Name × Int)) (value : TSyntax `tritonExpr) : Option Int :=
  match value with
  | `(tritonExpr| $n:num)=>some n.getNat
  | `(tritonExpr| $id:ident)=>(values.lookup id.getId.eraseMacroScopes)
  | `(tritonExpr| ($inner:tritonExpr))=>integer? values inner
  | `(tritonExpr| -$inner:tritonExpr)=>(-·) <$> integer? values inner
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)=>(·+·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)=>(·-·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)=>(·*·) <$> integer? values a <*> integer? values b
  | _=>none
private partial def boolean? (values : List (Name × Int)) (value : TSyntax `tritonExpr) : Option Bool :=
  match value with
  | `(tritonExpr| true)=>some Bool.true
  | `(tritonExpr| false)=>some Bool.false
  | `(tritonExpr| ($inner:tritonExpr))=>boolean? values inner
  | `(tritonExpr| not $inner:tritonExpr)=>(!·) <$> boolean? values inner
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr)=>(·&&·) <$> boolean? values a <*> boolean? values b
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr)=>(·||·) <$> boolean? values a <*> boolean? values b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr)=>(·==·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr)=>(·!=·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr)=>(·<·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr)=>(·<=·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr)=>(·>·) <$> integer? values a <*> integer? values b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr)=>(·>=·) <$> integer? values a <*> integer? values b
  | _=>none
private partial def select (values : List (Name × Int)) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let mut out:=#[]
  for statement in body do
    match statement with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>
      match boolean? values condition with
      | some value=>out:=out++(← select values (if value then yes else no))
      | none=>
        let yes ← select values yes
        let no ← select values no
        out:=out.push (← `(tritonStmt| if $condition { $yes* } else { $no* }))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* })=>
      match boolean? values condition with
      | some Bool.true=>out:=out++(← select values yes)
      | some Bool.false=>pure ()
      | none=>
        let yes ← select values yes
        out:=out.push (← `(tritonStmt| if $condition { $yes* }))
    | _=>out:=out.push statement
  return out

@[term_elab StatementJitFunctions.functionBody] def elaborateFunction : TermElab := fun stx expected=>do
  unless expected.any (·.isConstOf ``SpecializedStatementJitFunction) do throwUnsupportedSyntax
  let parameters:=stx[2].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let constants:=stx[6].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  unless parameters.eraseDups==parameters && constants.eraseDups==constants && constants.all parameters.contains do
    throwError "distinct helper parameters and a constexpr subset are required"
  let source ← text stx.getArgs.back!
  for statement in ← parse source do immutable constants statement
  let strings (names : List Name):=mkListLit (mkConst ``String) (names.map (mkStrLit ∘ Name.toString))
  ensureHasType expected (mkApp3 (mkConst ``SpecializedStatementJitFunction.mk)
    (← strings parameters) (← strings constants) (mkStrLit source))

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "static helper metadata required"
  return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "static helper names required"
  return text
private def names (value : Expr) : TermElabM (List Name) := do
  (← listValues value).mapM (fun value=>return (← stringValue value).toName)
private def expression (value : Int) : MacroM (TSyntax `tritonExpr) := do
  let numeral : TSyntax `num:=⟨Syntax.mkNumLit (toString value.natAbs)⟩
  if value<0 then `(tritonExpr| -$numeral:num) else `(tritonExpr| $numeral:num)

private partial def renameUnderscore (name : Name) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes==`_ then mkIdent name
  else raw.setArgs (raw.getArgs.map (renameUnderscore name))

private structure Function where
  name : Name
  parameters : List Name
  constants : List Name
  source : String

@[term_elab StatementJitFunctions.withFunctions] def elaborateInvocation : TermElab := fun stx expected=>do
  let mut functions : List Function:=[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding | throwUnsupportedSyntax
    let value ← elabTerm value none
    unless (← inferType value).isConstOf ``SpecializedStatementJitFunction do throwUnsupportedSyntax
    functions:=functions++[⟨id.getId.eraseMacroScopes,
      ← names (← mkAppM ``SpecializedStatementJitFunction.parameters #[value]),
      ← names (← mkAppM ``SpecializedStatementJitFunction.constexpr #[value]),
      ← stringValue (← mkAppM ``SpecializedStatementJitFunction.source #[value])⟩]
  if functions.isEmpty then throwUnsupportedSyntax
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let constants ← names (← mkAppM ``CallSignature.constexpr #[profile])
  let mut values : List (Name × Int):=[]
  for name in constants do
    let value ← elabTerm (mkIdent name) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then
      if let .lit (.natVal n) ← whnf value then values:=values++[(name,(n : Int))]
  let body ← parse (← text stx.getArgs.back!)
  let used:=body.toList.flatMap (ids ∘ TSyntax.raw)
  let mut underscore:=Name.mkSimple "__veritile_python_underscore"
  let mut suffix:=0
  while used.contains underscore do
    suffix:=suffix+1
    underscore:=Name.mkSimple ("__veritile_python_underscore_"++toString suffix)
  let body:=body.map (fun statement=>(⟨renameUnderscore underscore statement.raw⟩ : TSyntax `tritonStmt))
  for statement in body do immutable constants statement
  let mut out:=#[]
  let mut bindings : Array (TSyntax `tritonCallBinding):=#[]
  let used:=ids stx++body.toList.flatMap (ids ∘ TSyntax.raw)++functions.map (·.name)
  for statement in body do
    let call ← match statement with
      | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $callee:ident($args:statementJitArgument,*))=>
        pure (some (callee,args.getElems))
      | `(tritonStmt| $a:ident = $callee:ident($args:statementJitArgument,*))=>pure (some (callee,args.getElems))
      | _=>pure none
    if let some (callee,args):=call then
      if let some f:=functions.find? (·.name==callee.getId.eraseMacroScopes) then
        let mut fixed : List (Name × Int):=[]
        let mut normalized:=#[]
        let mut positional:=0
        for argument in args do
          let (parameter,value,key) ← match argument with
            | `(statementJitArgument| $key:ident = $value:tritonExpr)=>pure (key.getId.eraseMacroScopes,value,some key)
            | `(statementJitArgument| $value:tritonExpr)=>do
              let some parameter:=f.parameters[positional]? | throwError "too many JIT arguments"
              positional:=positional+1
              pure (parameter,value,none)
            | _=>throwError "unsupported JIT argument"
          let mut value:=value
          if f.constants.contains parameter then
            if let some n:=integer? values value then
              fixed:=fixed++[(parameter,n)]
              value ← liftMacroM (expression n)
          normalized:=normalized.push (← liftMacroM (match key with
            | some key=>`(statementJitArgument| $key:ident = $value:tritonExpr)
            | none=>`(statementJitArgument| $value:tritonExpr)))
        let selected ← liftMacroM (select fixed (← parse f.source))
        let source:=String.intercalate "\n" (selected.toList.map (render ∘ TSyntax.raw))
        let strings (ns : List Name):=mkListLit (mkConst ``String) (ns.map (mkStrLit ∘ Name.toString))
        let value:=mkApp3 (mkConst ``StatementJitFunction.mk) (← strings f.parameters)
          (← strings f.constants) (mkStrLit source)
        let value ← exprToSyntax value
        let mut name:=Name.mkSimple ("__veritile_specialized_jit_"++toString bindings.size)
        let mut helperSuffix:=0
        while used.contains name do
          helperSuffix:=helperSuffix+1
          name:=Name.mkSimple ("__veritile_specialized_jit_"++toString bindings.size++"_"++toString helperSuffix)
        let id:=mkIdent name
        bindings:=bindings.push (← `(tritonCallBinding| $id:ident := $value:term))
        let rewritten ← match statement with
          | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $callee:ident($_:statementJitArgument,*))=>
            `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $id:ident($normalized,*))
          | `(tritonStmt| $a:ident = $callee:ident($_:statementJitArgument,*))=>
            `(tritonStmt| $a:ident = $id:ident($normalized,*))
          | _=>throwError "unsupported specialized JIT destination"
        out:=out.push rewritten
        continue
    out:=out.push statement
  let profileSyntax : TSyntax `term:=⟨stx[1]⟩
  let callBindings:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let base ← `(with_triton_call $profileSyntax:term call_args [$callBindings,*]
    jit_functions [$bindings,*] in triton {})
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (out.toList.map (render ∘ TSyntax.raw))++"\n}")
  GuardedJitFunctions.elaborateInvocation (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)) expected
end DSL.SpecializedJitFunctions
end VeriTile.Triton
