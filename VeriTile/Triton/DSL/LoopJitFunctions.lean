import VeriTile.Triton.DSL.StatementJitFunctions

/-! Optional complete-source JIT helper frontend for counted loops and tuple
returns. Arguments are captured once, locals are renamed, and every returned
expression is evaluated before caller destinations are changed. Only declared
caller constexpr names are substituted directly; symbolic scalar/tensor values
are captured using the same typed expression elaborator as ordinary arguments.
The original helper and caller bodies remain verbatim source quotations. -/
namespace VeriTile.Triton.DSL.LoopJitFunctions
open Lean Elab Term Meta
syntax (name := tupleValues) ident "," ident "=" tritonExpr "," tritonExpr : tritonStmt

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
  if raw.getKind==choiceKind then firstChoice raw[0]! else raw.setArgs (raw.getArgs.map firstChoice)

private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let text := PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let raw ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body => pure body
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := raw | throwError "expected a complete JIT body"
  return stmts.map fun stmt=>⟨firstChoice stmt.raw⟩

private def tuple? (stmt : TSyntax `tritonStmt) :
    Option (Ident × Ident × TSyntax `tritonExpr × TSyntax `tritonExpr) :=
  if stmt.raw.getKind==``tupleValues && stmt.raw.getNumArgs==7 then
    some (⟨stmt.raw[0]⟩,⟨stmt.raw[2]⟩,⟨stmt.raw[4]⟩,⟨stmt.raw[6]⟩)
  else none

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
        spelling.startsWith "tl." || ["float","True","False","None"].contains spelling do
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
  let bound:=parameters
  let immutable:=constants
  let locals:List Name:=[]
  let rec validateStatements (fuel : Nat) (stmts : Array (TSyntax `tritonStmt))
      (bound immutable locals : List Name) :
      TermElabM (List Name × List Name × List Name) := do
    let fuel+1:=fuel | throwError "JIT source nesting exceeds 128 levels"
    let mut bound:=bound
    let mut immutable:=immutable
    let mut locals:=locals
    for stmt in stmts do
      if let `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* }) := stmt then
        for expression in [lo,hi,stride] do checkExpression bound expression
        let name:=counter.getId.eraseMacroScopes
        unless simpleName name && !immutable.contains name do
          throwErrorAt counter "a JIT loop cannot bind a reserved name or constexpr"
        if !bound.contains name then bound:=bound++[name];locals:=locals++[name]
        let (b,i,l) ← validateStatements fuel nested bound immutable locals
        bound:=b;immutable:=i;locals:=l
      else if let some (left,right,a,b):=tuple? stmt then
        checkExpression bound a;checkExpression bound b
        for id in [left,right] do
          let name:=id.getId.eraseMacroScopes
          unless simpleName name && !immutable.contains name do
            throwErrorAt id "a JIT tuple cannot rebind a constexpr or reserved name"
          if !bound.contains name then bound:=bound++[name];locals:=locals++[name]
      else if let some (id,rhs,constant):=assignment? stmt then
        let name:=id.getId.eraseMacroScopes
        unless simpleName name do throwErrorAt id "reserved JIT local name"
        if immutable.contains name || (constant && bound.contains name) then
          throwErrorAt id "a JIT constexpr cannot be rebound"
        checkExpression bound rhs
        if constant then immutable:=immutable++[name]
        if !bound.contains name then bound:=bound++[name];locals:=locals++[name]
      else
        let args:=stmt.raw.getArgs
        unless args.size==3 && args[0]!.isIdent && args[1]!.isAtom &&
            ["+=","-=","*=","/="].contains args[1]!.getAtomVal.trimAscii.toString do
          throwErrorAt stmt "unsupported loop JIT statement ({stmt.raw.getKind}, {args.size} arguments): {PythonFullConstructors.render stmt.raw}"
        let name:=args[0]!.getId.eraseMacroScopes
        unless bound.contains name && !immutable.contains name do
          throwErrorAt stmt "an augmented JIT assignment needs a mutable existing binding"
        checkExpression bound args[2]!
    return (bound,immutable,locals)
  let (bound,_,locals) ← validateStatements 128 body bound immutable locals
  for value in results do checkExpression bound value
  return (body,results,locals)

@[term_elab StatementJitFunctions.functionBody] def elaborateFunction : TermElab := fun stx expected => do
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
    let directConstant := match value with
      | `(tritonExpr| $id:ident) => f.constants.contains name && constants.contains id.getId.eraseMacroScopes
      | _ => Bool.false
    if directConstant then
      bindings:=bindings++[(name,value.raw)]
    else
      let localName ← fresh used
      result:=result.push (← `(tritonStmt| $localName:ident = $value:tritonExpr))
      bindings:=bindings++[(name,localName.raw)]
  for localName in f.locals do
    bindings:=bindings++[(localName,(← fresh used).raw)]
  for stmt in f.body do
    let stmt:TSyntax `tritonStmt:=⟨rename bindings stmt⟩
    if let some (left,right,a,b):=tuple? stmt then
      let av ← fresh used
      let bv ← fresh used
      result:=result.push (← `(tritonStmt| $av:ident = $a:tritonExpr))
      result:=result.push (← `(tritonStmt| $bv:ident = $b:tritonExpr))
      result:=result.push (← `(tritonStmt| $left:ident = $av:ident))
      result:=result.push (← `(tritonStmt| $right:ident = $bv:ident))
    else result:=result.push stmt
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

private def dtypeValue (value : Expr) : TermElabM Triton.SourceDType := do
  let value ← whnf value
  let kinds : List (Name × Triton.SourceDType) :=
    [(``Triton.SourceDType.fp16,.fp16),(``Triton.SourceDType.bf16,.bf16),
     (``Triton.SourceDType.fp32,.fp32),(``Triton.SourceDType.fp64,.fp64),
     (``Triton.SourceDType.f8e4,.f8e4),(``Triton.SourceDType.f8e5,.f8e5),
     (``Triton.SourceDType.i8,.i8),(``Triton.SourceDType.i16,.i16),
     (``Triton.SourceDType.i32,.i32),(``Triton.SourceDType.i64,.i64),
     (``Triton.SourceDType.u8,.u8),(``Triton.SourceDType.u16,.u16),
     (``Triton.SourceDType.u32,.u32),(``Triton.SourceDType.u64,.u64),
     (``Triton.SourceDType.boolean,.boolean)]
  if let some (_,kind) := kinds.find? (fun p => value.isConstOf p.1) then return kind
  throwError "call signature dtypes must be statically known"

private partial def elementQueries (raw : Syntax) : List Name :=
  let own:=if raw.isIdent then match raw.getId.eraseMacroScopes with
    | .str (.str pointer "type") "element_ty"
    | .str (.str pointer "dtype") "element_ty" => [pointer]
    | _=>[]
    else []
  own++raw.getArgs.toList.flatMap elementQueries

private partial def lowerElements (values : List (Name × Syntax)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    match raw.getId.eraseMacroScopes with
    | .str (.str pointer "type") "element_ty"
    | .str (.str pointer "dtype") "element_ty" =>
      if let some (_,dtype):=values.find? (·.1==pointer) then dtype else raw
    | _=>raw
  else raw.setArgs (raw.getArgs.map (lowerElements values))

@[term_elab StatementJitFunctions.withFunctions] def elaborateInvocation : TermElab := fun stx expected => do
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
  let used:=occupied++ids stx++body.toList.flatMap (ids ∘ TSyntax.raw)++
    functions.flatMap (fun f=>f.parameters++f.locals)
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
  let queries:List Name:=(lowered.toList.flatMap (elementQueries ∘ TSyntax.raw)).eraseDups
  let mut elements:List (Name × Syntax):=[]
  for pointer in queries do
    for stmt in body do immutableArguments [pointer] stmt
    let mut found:=Bool.false
    for pair in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
      let name ← stringValue (← mkAppM ``Prod.fst #[pair])
      if name.toName==pointer then
        let dtype ← dtypeValue (← mkAppM ``Prod.snd #[pair])
        let some dt ← liftMacroM (dtypeSyntaxOfName? (SourceDType.dtypeName dtype))
          | throwError "unsupported pointer element dtype"
        elements:=elements++[(pointer,dt.raw)]
        found:=Bool.true
    unless found do throwError "element dtype query needs an immutable statically typed pointer: {pointer}"
  lowered:=lowered.map fun stmt=>(⟨lowerElements elements stmt.raw⟩ : TSyntax `tritonStmt)
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (lowered.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let base:=mkNode ``CallSignature.withCall
    #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
  PythonFullConstructors.elaborateInvocation base expected

end VeriTile.Triton.DSL.LoopJitFunctions
