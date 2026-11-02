import VeriTile.Triton.DSL.PythonFullConstructors

/-! Closed, straight-line Triton helpers with ordinary positional/keyword
arguments and one scalar or tuple return. Helpers retain their complete pasted
bodies. Expansion captures runtime arguments once, renames every local, and
evaluates every returned expression before assigning any caller destination.
Constexpr arguments are substituted only from declared caller constexprs.
Unsupported control flow, free variables and escaping calls are rejected. -/
namespace VeriTile.Triton

structure StatementJitFunction where
  parameters : List String
  constexpr : List String
  source : String

namespace DSL.StatementJitFunctions
open Lean Elab Term Meta

declare_syntax_cat statementJitArgument
syntax tritonExpr : statementJitArgument
syntax ident "=" tritonExpr : statementJitArgument
syntax (name := helperCall) (priority := low) ident "(" statementJitArgument,* ")" : tritonExpr
syntax (name := helperTupleCall) ident "," ident ("," ident)* "="
  ident "(" statementJitArgument,* ")" : tritonStmt
syntax (name := helperReturn) "return " tritonExpr,* : tritonStmt
syntax (name := functionBody) "triton_function " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" RawSourceText.body : term
syntax (name := withFunctions) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " jit_functions " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term

private def sourceText (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok source => pure source
  | .error message => throwErrorAt raw "{message}"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap ids

private def simpleName (name : Name) : Bool :=
  name.isStr && name.getPrefix.isAnonymous &&
    !["tl","triton","libdevice","None","True","False"].contains name.toString

private partial def firstChoice (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then firstChoice raw[0]! else raw

private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let text := PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let raw ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body => pure body
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := raw | throwError "expected a complete JIT body"
  return stmts.map fun stmt=>⟨firstChoice stmt.raw⟩

private def assignment? (stmt : TSyntax `tritonStmt) : Option (Ident × TSyntax `tritonExpr × Bool) :=
  match stmt with
  | `(tritonStmt| $id:ident = $value:tritonExpr) => some (id,value,Bool.false)
  | `(tritonStmt| $id:ident : tl.constexpr = $value:tritonExpr) => some (id,value,Bool.true)
  | _ => none

private partial def checkExpression (bound : List Name) (raw : Syntax) : TermElabM Unit := do
  if raw.getKind==choiceKind then return ← checkExpression bound raw[0]!
  if raw.isAtom && raw.getAtomVal.contains '$' then
    throwErrorAt raw "Lean expressions cannot occur in a pasted JIT helper"
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    let spelling:=name.toString
    let root:=(spelling.splitOn ".").head!.toName
    unless bound.contains name || (bound.contains root && spelling.endsWith ".to") ||
        spelling.startsWith "tl." do
      throwErrorAt raw "JIT helper references an unbound name: {name}"
  let args:=raw.getArgs
  for i in [:args.size] do
    -- An intrinsic keyword is syntax, not a reference to a caller local.
    let key:=args[i]!.isIdent && args[i+1]?.any
      (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString=="=")
    unless key do checkExpression bound args[i]!

private def validate (parameters constants : List Name) (source : String) :
    TermElabM (Array (TSyntax `tritonStmt) × Array (TSyntax `tritonExpr) × List Name) := do
  unless !parameters.isEmpty && parameters.all simpleName && parameters.eraseDups==parameters do
    throwError "JIT helper parameters must be distinct unreserved simple names"
  unless constants.eraseDups==constants && constants.all parameters.contains do
    throwError "JIT helper constexpr parameters must be a distinct subset of its parameters"
  let stmts ← parse source
  let some last:=stmts.back? | throwError "JIT helper needs a final return"
  let `(tritonStmt| return $results:tritonExpr,*) := last
    | throwError "JIT helper needs one final scalar or tuple return"
  let results:=results.getElems
  unless !results.isEmpty do throwError "empty JIT returns are unsupported"
  let body:=stmts.pop
  let mut bound:=parameters
  let mut immutable:=constants
  let mut locals:List Name:=[]
  for stmt in body do
    let some (id,rhs,constant):=assignment? stmt
      | throwErrorAt stmt "statement JIT helpers currently require straight-line assignments"
    let name:=id.getId.eraseMacroScopes
    unless simpleName name do throwErrorAt id "reserved JIT local name"
    if immutable.contains name || (constant && bound.contains name) then
      throwErrorAt id "a JIT constexpr cannot be rebound"
    checkExpression bound rhs
    if constant then immutable:=immutable++[name]
    if !bound.contains name then
      bound:=bound++[name]
      locals:=locals++[name]
  for value in results do checkExpression bound value
  return (body,results,locals)

@[term_elab functionBody] def elaborateFunction : TermElab := fun stx expected => do
  let parameters:=stx[2].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let constants:=stx[6].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let source ← sourceText stx.getArgs.back!
  let _ ← validate parameters constants source
  let strings (names : List Name) := mkListLit (mkConst ``String) (names.map (mkStrLit ∘ Name.toString))
  let value:=mkApp3 (mkConst ``StatementJitFunction.mk) (← strings parameters)
    (← strings constants) (mkStrLit source)
  return ← ensureHasType expected value

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "JIT source and names must be static"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "JIT function metadata must be static"
private def names (value : Expr) : TermElabM (List Name) := do
  (← listValues value).mapM fun value=>return (← stringValue value).toName

private structure Function where
  name : Name
  parameters : List Name
  constants : List Name
  body : Array (TSyntax `tritonStmt)
  results : Array (TSyntax `tritonExpr)
  locals : List Name

private partial def rename (bindings : List (Name × Syntax)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    match bindings.find? (·.1==name) with
    | some (_,value) => value
    | none =>
      let spelling:=name.toString
      if spelling.endsWith ".to" then
        let root:=(spelling.dropEnd 3).toString.toName
        match bindings.find? (·.1==root) with
        | some (_,value) => mkIdent (value.getId.eraseMacroScopes.toString++".to").toName
        | none => raw
      else raw
  else Id.run do
    let args:=raw.getArgs
    let leaf:=match raw.getKind with | .str _ name=>name | _=>""
    let keywordOwner:=leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
      leaf.startsWith "tritonReduceKwarg" || leaf.startsWith "tritonKwarg"
    return raw.setArgs (args.mapIdx fun i arg=>
      if keywordOwner && arg.isIdent && args[i+1]?.any
          (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString=="=")
      then arg else rename bindings arg)

private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_statement_jit_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_statement_jit_"++toString n))

private def expand (f : Function) (constants used : List Name)
    (arguments : Array (TSyntax `statementJitArgument)) (destinations : Array Ident) :
    StateRefT Nat TermElabM (Array (TSyntax `tritonStmt)) := do
  unless destinations.size==f.results.size do throwError "JIT return arity does not match assignment"
  let mut args:List (Name × TSyntax `tritonExpr):=[]
  let mut keyword:=Bool.false
  let mut positional:=0
  for arg in arguments do
    let (name,value) ← match arg with
      | `(statementJitArgument| $key:ident = $value:tritonExpr) => do
        keyword:=Bool.true
        pure (key.getId.eraseMacroScopes,value)
      | `(statementJitArgument| $value:tritonExpr) => do
        if keyword then throwError "a positional JIT argument cannot follow a keyword"
        let some name:=f.parameters[positional]? | throwError "too many positional JIT arguments"
        positional:=positional+1
        pure (name,value)
      | _ => throwError "unsupported JIT argument"
    unless f.parameters.contains name do throwError "unknown JIT keyword argument: {name}"
    if args.any (·.1==name) then throwError "duplicate JIT argument: {name}"
    args:=args++[(name,value)]
  unless args.length==f.parameters.length do throwError "missing JIT arguments"
  let mut bindings:List (Name × Syntax):=[]
  let mut result:=#[]
  -- Preserve Python's actual argument evaluation order, including keywords.
  for (name,value) in args do
    if f.constants.contains name then
      let `(tritonExpr| $id:ident) := value
        | throwError "a helper constexpr currently requires a declared caller constexpr name"
      unless constants.contains id.getId.eraseMacroScopes do
        throwError "a helper constexpr cannot receive a runtime register"
      bindings:=bindings++[(name,id.raw)]
    else
      let localName ← fresh used
      result:=result.push (← `(tritonStmt| $localName:ident = $value:tritonExpr))
      bindings:=bindings++[(name,localName.raw)]
  for localName in f.locals do
    bindings:=bindings++[(localName,(← fresh used).raw)]
  for stmt in f.body do result:=result.push ⟨rename bindings stmt⟩
  let mut returned:Array Ident:=#[]
  for value in f.results do
    let localName ← fresh used
    let value:TSyntax `tritonExpr:=⟨rename bindings value⟩
    result:=result.push (← `(tritonStmt| $localName:ident = $value:tritonExpr))
    returned:=returned.push localName
  for i in [:destinations.size] do
    let destination:=destinations[i]!
    let value:=returned[i]!
    result:=result.push (← `(tritonStmt| $destination:ident = $value:ident))
  return result

private partial def noEscapes (functions : List Name) (raw : Syntax) : TermElabM Unit := do
  if (ids raw).any functions.contains then
    throwErrorAt raw "JIT helpers may occur only in direct, plain assignments"

private partial def immutableArguments (constants : List Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs.toList
  if args.head?.any (·.isIdent) then
    if let some i:=args.findIdx? (fun x=>x.isAtom &&
        ["=",":=","+=","-=","*=","/="].contains x.getAtomVal.trimAscii.toString) then
      if ((args.take i).flatMap ids).any constants.contains then
        throwError "caller constexpr arguments must remain immutable"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any constants.contains then
      throwError "a loop cannot shadow a caller constexpr argument"
  for child in raw.getArgs do immutableArguments constants child

@[term_elab withFunctions] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let constants ← names (← mkAppM ``CallSignature.constexpr #[profile])
  let mut occupied:=constants
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      occupied:=occupied++[(← stringValue (← mkAppM ``Prod.fst #[pair])).toName]
  unless occupied.eraseDups==occupied do throwError "duplicate call-signature argument"
  let mut functions:List Function:=[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding | throwError "named JIT helper required"
    let name:=id.getId.eraseMacroScopes
    unless simpleName name && !occupied.contains name && !(functions.any (·.name==name)) do
      throwError "JIT helper name collides with a namespace, argument or another helper"
    let value ← elabTermEnsuringType value (mkConst ``StatementJitFunction)
    let parameters ← names (← mkAppM ``StatementJitFunction.parameters #[value])
    let helperConstants ← names (← mkAppM ``StatementJitFunction.constexpr #[value])
    let source ← stringValue (← mkAppM ``StatementJitFunction.source #[value])
    let (body,results,locals) ← validate parameters helperConstants source
    functions:=functions++[⟨name,parameters,helperConstants,body,results,locals⟩]
  let body ← parse (← sourceText stx.getArgs.back!)
  for stmt in body do immutableArguments constants stmt
  let used:=occupied++ids stx++body.toList.flatMap (ids ∘ TSyntax.raw)
  let mut lowered:=#[]
  let mut next:=0
  for stmt in body do
    let call ← match stmt with
      | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $callee:ident($args:statementJitArgument,*)) =>
        pure (some (#[a,b]++more,callee,args.getElems))
      | `(tritonStmt| $destination:ident = $callee:ident($args:statementJitArgument,*)) =>
        pure (some (#[destination],callee,args.getElems))
      | `(tritonStmt| $a:ident, $b:ident = $callee:ident($args:tritonExpr,*)) => do
        let args ← args.getElems.mapM fun value=>liftMacroM `(statementJitArgument| $value:tritonExpr)
        pure (some (#[a,b],callee,args))
      | `(tritonStmt| $a:ident, $b:ident, $c:ident = $callee:ident($args:tritonExpr,*)) => do
        let args ← args.getElems.mapM fun value=>liftMacroM `(statementJitArgument| $value:tritonExpr)
        pure (some (#[a,b,c],callee,args))
      | _ => pure none
    if let some (destinations,callee,args):=call then
      if let some f:=functions.find? (·.name==callee.getId.eraseMacroScopes) then
        for value in args do noEscapes (functions.map (·.name)) value
        for destination in destinations do noEscapes (functions.map (·.name)) destination
        let (expanded,n) ← (expand f constants used args destinations).run next
        next:=n
        lowered:=lowered++expanded
        continue
    noEscapes (functions.map (·.name)) stmt
    lowered:=lowered.push stmt
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (lowered.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let base:=mkNode ``CallSignature.withCall
    #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
  PythonFullConstructors.elaborateInvocation base expected

end DSL.StatementJitFunctions
end VeriTile.Triton
