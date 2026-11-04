import VeriTile.Triton.DSL.GuardedJitFunctions
import VeriTile.Triton.DSL.SignedAugmentedAssignments

/-! Opt-in recursive statement JIT calls. This extends GuardedJitFunctions with
calls inside source branches and ranges, statement-only write checks, and
integer augmented-assignment promotion. Its guarded helper validation and
capture-once lowering follow the existing frontend. The original source body
is retained in each quotation; expansion is solely a frontend operation. -/
namespace VeriTile.Triton.DSL.RecursiveJitFunctions
open Lean Elab Term Meta
syntax (name := tupleValues) ident "," ident "=" tritonExpr "," tritonExpr : tritonStmt

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let leaf:=match raw.getKind with | .str _ name=>name | _=>""
  if leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
      leaf.startsWith "tritonReduceKwarg" || leaf.startsWith "tritonKwarg" then return

  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x=>x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x=>(ids x).any names.contains) then
        throwError "integer JIT constexpr arguments must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then
      throwError "a loop cannot shadow an integer JIT constexpr argument"
  for child in raw.getArgs do rejectWrites names child

private partial def natural? (values : List (Name × Nat)) (value : TSyntax `tritonExpr) : Option Nat :=
  match value with
  | `(tritonExpr| $n:num) => some n.getNat
  | `(tritonExpr| $id:ident) => (values.find? (·.1==id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| ($value:tritonExpr)) => natural? values value
  | _ => none

private partial def boolean? (values : List (Name × Nat)) (value : TSyntax `tritonExpr) : Option Bool :=
  match value with
  | `(tritonExpr| true) => some Bool.true
  | `(tritonExpr| false) => some Bool.false
  | `(tritonExpr| ($value:tritonExpr)) => boolean? values value
  | `(tritonExpr| not $value:tritonExpr) => (!·) <$> boolean? values value
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => (· && ·) <$> boolean? values a <*> boolean? values b
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) => (· || ·) <$> boolean? values a <*> boolean? values b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => (· == ·) <$> natural? values a <*> natural? values b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => (· != ·) <$> natural? values a <*> natural? values b
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => (· < ·) <$> natural? values a <*> natural? values b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => (· <= ·) <$> natural? values a <*> natural? values b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => (· > ·) <$> natural? values a <*> natural? values b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => (· >= ·) <$> natural? values a <*> natural? values b
  | _ => none

private partial def select (values : List (Name × Nat))
    (stmts : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match stmts with
  | [] => return []
  | stmt::rest =>
    match stmt with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if let some selected := boolean? values condition then
        return ← select values ((if selected then yes else no).toList++rest)
      let yes ← select values yes.toList
      let no ← select values no.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* } else { $no.toArray* })
      return stmt::(← select values rest)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if let some selected := boolean? values condition then
        return ← select values ((if selected then yes.toList else [])++rest)
      let yes ← select values yes.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* })
      return stmt::(← select values rest)
    | _ => return stmt::(← select values rest)


/-- Return Python spellings for identifiers which the raw parser escapes.
The next raw-body parser performs its own escaping exactly once. -/
private partial def pythonTokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling name _ =>
    let name:=name.eraseMacroScopes.toString
    [if name=="_" || name=="do" then name else spelling.toString]
  | .node _ kind args => if kind==choiceKind then pythonTokens args[0]! else args.toList.flatMap pythonTokens
  | .missing => []
private def renderPython (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in pythonTokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

private def sourceText (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok source => pure source
  | .error message => throwErrorAt raw "{message}"

private def simpleName (name : Name) : Bool :=
  name.isStr && name.getPrefix.isAnonymous &&
    !["tl","triton","libdevice","None","True","False"].contains name.toString

private partial def firstChoice (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then firstChoice raw[0]! else raw.setArgs (raw.getArgs.map firstChoice)

private partial def normalizeRanges (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM normalizeRanges)
  match raw with
  | `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr, $stride:tritonExpr) { $body:tritonStmt* }) =>
    return (← `(tritonStmt| for $counter:ident in range((0), $hi:tritonExpr, $stride:tritonExpr) { $body* })).raw
  | `(tritonStmt| for $counter:ident in range(0, $hi:tritonExpr) { $body:tritonStmt* }) =>
    return (← `(tritonStmt| for $counter:ident in range((0), $hi:tritonExpr, 1) { $body* })).raw
  | `(tritonStmt| for $counter:ident in range($hi:tritonExpr) { $body:tritonStmt* }) =>
    return (← `(tritonStmt| for $counter:ident in range((0), $hi:tritonExpr, 1) { $body* })).raw
  | _ => return raw

/-- Both positional-full parsers have the same grammar. Normalize their
nodes before selecting a parser alternative, retaining the declared dtype. -/
private partial def fullConstructors (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM fullConstructors)
  if [``FloatDTypeConstants.fullPositionalType,``PythonFullConstructors.positionalFull].contains raw.getKind then
    unless raw.getNumArgs==9 do Macro.throwErrorAt raw "invalid positional full constructor"
    let dims: Array (TSyntax `tritonExpr):=raw[2].getSepArgs.map (⟨·⟩)
    let value : TSyntax `tritonExpr:=⟨raw[5]⟩
    let dtype : TSyntax `tritonDType:=⟨raw[7]⟩
    return (← `(tritonExpr| tl.full([$dims,*], $value:tritonExpr, dtype=$dtype:tritonDType))).raw
  return raw

/-- A qualified `p.to` is also parsed as a generic helper call. The source
normalizer can retain that alternative beside the correctly parenthesized
cast. Select the typed cast rather than sending an unresolved choice to the
expression elaborator. No conversion node is removed or specialized. -/
private partial def castAlternatives (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then
    let typed:=raw.getArgs.find? fun candidate=>
      [``tritonMethodCastElementTy,``tritonMethodCastDTypeIdent,
       ``tritonMethodCast,``tritonIdentMethodCastDTypeIdent,
       ``tritonIdentMethodCastDTypeIdentSpaced].contains candidate.getKind
    match typed with
    | some cast => castAlternatives cast
    | none => raw.setArgs (raw.getArgs.map castAlternatives)
  else raw.setArgs (raw.getArgs.map castAlternatives)

@[macro tritonBlock]
def chooseTypedCasts : Macro := fun source=>do
  let prepared:=castAlternatives source
  if prepared==source then Macro.throwUnsupported
  return prepared

private def parse (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let text := PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let raw ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body => pure body
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := raw | throwError "expected a complete JIT body"
  liftMacroM <| stmts.mapM fun stmt=>return ⟨← fullConstructors (← normalizeRanges (firstChoice stmt.raw))⟩

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
    unless bound.contains name || (bound.contains root && [".to",".dtype",".dtype.element_ty",".type.element_ty"].any (fun suffix=>spelling.endsWith suffix)) ||
        spelling.startsWith "tl." || ["float","True","False","None"].contains spelling do
      throwErrorAt raw "JIT helper references an unbound name: {name}"
  let args:=raw.getArgs
  for i in [:args.size] do
    -- An intrinsic keyword is syntax, not a reference to a caller local.
    let key:=args[i]!.isIdent && args[i+1]?.any
      (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString=="=")
    unless key do checkExpression bound args[i]!

private abbrev GuardPath := List (String × Bool)
private abbrev Definitions := List (Name × List GuardPath)

private def holds (atoms : List String) (assignment : Nat) (path : GuardPath) : Bool :=
  path.all fun (key,value)=>
    match atoms.idxOf? key with
    | some index => ((assignment/2^index)%2==1)==value
    | none => Bool.false

private def available (definitions : Definitions) (path : GuardPath) : TermElabM (List Name) := do
  let mut names:=[]
  for (name,alternatives) in definitions do
    let atoms:=((path.map Prod.fst)++alternatives.flatMap (List.map Prod.fst)).eraseDups
    unless atoms.length≤12 do throwError "JIT definite-assignment guard exceeds 12 independent tests"
    if (List.range (2^atoms.length)).all (fun assignment=>
        !(holds atoms assignment path) || alternatives.any (holds atoms assignment)) then
      names:=names++[name]
  return names

private def addDefinition (definitions : Definitions) (name : Name) (path : GuardPath) : Definitions :=
  (name,(path::(definitions.lookup name).getD []).eraseDups)::definitions.filter (·.1!=name)

private def unionDefinitions (left right : Definitions) : Definitions :=
  right.foldl (fun definitions (name,paths)=>paths.foldl (fun definitions path=>
    addDefinition definitions name path) definitions) left

private def guardKey (immutable : List Name) (condition : TSyntax `tritonExpr) : String :=
  let key:=PythonFullConstructors.render condition.raw
  if (ids condition.raw).all immutable.contains then key
  else key++"@runtime:"++reprStr condition.raw.getHeadInfo

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
  let rec validateStatements (fuel : Nat) (stmts : Array (TSyntax `tritonStmt))
      (definitions : Definitions) (path : GuardPath) (immutable locals : List Name) :
      TermElabM (Definitions × List Name × List Name) := do
    let fuel+1:=fuel | throwError "JIT source nesting exceeds 128 levels"
    let mut definitions:=definitions
    let mut immutable:=immutable
    let mut locals:=locals
    for stmt in stmts do
      let bound ← available definitions path
      if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) := stmt then
        checkExpression bound condition
        let key:=guardKey immutable condition
        let (yd,yi,yl) ← validateStatements fuel yes definitions ((key,Bool.true)::path) immutable locals
        let (nd,ni,nl) ← validateStatements fuel no definitions ((key,Bool.false)::path) immutable locals
        definitions:=unionDefinitions yd nd
        immutable:=(yi++ni).eraseDups
        locals:=(yl++nl).eraseDups
      else if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) := stmt then
        checkExpression bound condition
        let key:=guardKey immutable condition
        let (yd,yi,yl) ← validateStatements fuel yes definitions ((key,Bool.true)::path) immutable locals
        definitions:=unionDefinitions definitions yd
        immutable:=(immutable++yi).eraseDups
        locals:=(locals++yl).eraseDups
      else if let `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* }) := stmt then
        for expression in [lo,hi,stride] do checkExpression bound expression
        let name:=counter.getId.eraseMacroScopes
        unless simpleName name && !immutable.contains name do
          throwErrorAt counter "a JIT loop cannot bind a reserved name or constexpr"
        if !(definitions.any (·.1==name)) then locals:=locals++[name]
        let (_,i,l) ← validateStatements fuel nested (addDefinition definitions name path) path immutable locals
        -- A zero-trip loop defines no new values outside its body.
        immutable:=i;locals:=l
      else if let some (left,right,a,b):=tuple? stmt then
        checkExpression bound a;checkExpression bound b
        for id in [left,right] do
          let name:=id.getId.eraseMacroScopes
          unless simpleName name && !immutable.contains name do
            throwErrorAt id "a JIT tuple cannot rebind a constexpr or reserved name"
          if !(definitions.any (·.1==name)) then locals:=locals++[name]
          definitions:=addDefinition definitions name path
      else if let some (id,rhs,constant):=assignment? stmt then
        let name:=id.getId.eraseMacroScopes
        unless simpleName name do throwErrorAt id "reserved JIT local name"
        if immutable.contains name || (constant && definitions.any (·.1==name)) then
          throwErrorAt id "a JIT constexpr cannot be rebound"
        checkExpression bound rhs
        if constant then immutable:=immutable++[name]
        if !(definitions.any (·.1==name)) then locals:=locals++[name]
        definitions:=addDefinition definitions name path
      else
        let args:=stmt.raw.getArgs
        unless args.size==3 && args[0]!.isIdent && args[1]!.isAtom &&
            ["+=","-=","*=","/="].contains args[1]!.getAtomVal.trimAscii.toString do
          throwErrorAt stmt "unsupported guarded JIT statement: {PythonFullConstructors.render stmt.raw}"
        let name:=args[0]!.getId.eraseMacroScopes
        unless bound.contains name && !immutable.contains name do
          throwErrorAt stmt "an augmented JIT assignment needs a definitely assigned mutable binding"
        checkExpression bound args[2]!
    return (definitions,immutable,locals)
  let initial : Definitions := parameters.map fun name=>(name,[[]])
  let (definitions,_,locals) ← validateStatements 128 body initial [] constants []
  let bound ← available definitions []
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
      let parts:=spelling.splitOn "."
      let root:=parts.head!.toName
      match bindings.find? (·.1==root) with
      | some (_,value) =>
        if parts.length>1 then mkIdent (value.getId.eraseMacroScopes.toString++"."++String.intercalate "." parts.tail!).toName
        else raw
      | none => raw
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

private def scopedLocal (scope : Nat) (used : List Name) (name : Name) : Ident := Id.run do
  let base:="__veritile_statement_jit_local_"++toString scope++"_"++name.toString
  let mut result:=base
  let mut suffix:=0
  while used.contains (Name.mkSimple result) do
    suffix:=suffix+1
    result:=base++"_"++toString suffix
  return mkIdent (Name.mkSimple result)

private partial def lowerTuples (used : List Name) (body : Array (TSyntax `tritonStmt)) :
    StateRefT Nat TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut result:=#[]
  for stmt in body do
    if let some (left,right,a,b):=tuple? stmt then
      let av ← fresh used
      let bv ← fresh used
      result:=result.push (← `(tritonStmt| $av:ident = $a:tritonExpr))
      result:=result.push (← `(tritonStmt| $bv:ident = $b:tritonExpr))
      result:=result.push (← `(tritonStmt| $left:ident = $av:ident))
      result:=result.push (← `(tritonStmt| $right:ident = $bv:ident))
    else if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) := stmt then
      let yes ← lowerTuples used yes
      let no ← lowerTuples used no
      result:=result.push (← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* }))
    else if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) := stmt then
      let yes ← lowerTuples used yes
      result:=result.push (← `(tritonStmt| if $condition:tritonExpr { $yes* }))
    else if let `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* }) := stmt then
      let nested ← lowerTuples used nested
      result:=result.push (← `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested* }))
    else result:=result.push stmt
  return result

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
      | `(tritonExpr| $_:num) => f.constants.contains name
      | _ => Bool.false
    if directConstant then
      bindings:=bindings++[(name,value.raw)]
    else
      let localName ← fresh used
      result:=result.push (← `(tritonStmt| $localName:ident = $value:tritonExpr))
      bindings:=bindings++[(name,localName.raw)]
  let scope ← get
  for localName in f.locals do
    let occupied:=used++bindings.flatMap (ids ∘ Prod.snd)
    let renamed:=scopedLocal scope occupied localName
    bindings:=bindings++[(localName,renamed.raw)]
  let renamed:=f.body.map fun statement=>(⟨rename bindings statement.raw⟩ : TSyntax `tritonStmt)
  result:=result++(← lowerTuples used renamed)
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
  let leaf:=match raw.getKind with | .str _ name=>name | _=>""
  if leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
      leaf.startsWith "tritonReduceKwarg" || leaf.startsWith "tritonKwarg" then return

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

private partial def checkPointerRebindings (pointer : Name) (raw : Syntax) : TermElabM Unit := do
  let args:=raw.getArgs
  if args.size≥2 && args[0]!.isIdent && args[0]!.getId.eraseMacroScopes==pointer &&
      args[1]!.isAtom && ["=",":=","*=","/="].contains args[1]!.getAtomVal.trimAscii.toString then
    throwError "pointer dtype query cannot cross a pointer rebinding"
  for child in args do checkPointerRebindings pointer child

private partial def lowerElements (values : List (Name × Syntax)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    match raw.getId.eraseMacroScopes with
    | .str (.str pointer "type") "element_ty"
    | .str (.str pointer "dtype") "element_ty" =>
      if let some (_,dtype):=values.find? (·.1==pointer) then dtype else raw
    | _=>raw
  else raw.setArgs (raw.getArgs.map (lowerElements values))

def withoutBooleanNames (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with constexpr:=profile.constexpr.filter (fun name=>!names.contains name)}

private partial def bindBooleans (values : List (Name × Syntax)) (raw : Syntax) : Syntax :=
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,value):=values.find? (·.1==id.getId.eraseMacroScopes) then value else raw
  else raw.setArgs (raw.getArgs.map (bindBooleans values))

private partial def expandNested (functions : List Function) (constants used : List Name)
    (body : Array (TSyntax `tritonStmt)) (next : Nat) :
    TermElabM (Array (TSyntax `tritonStmt) × Nat) := do
  let mut next:=next
  let mut lowered:=#[]
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
    let stmt ← match stmt with
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>do
        noEscapes (functions.map (·.name)) condition
        let (yes,n) ← expandNested functions constants used yes next
        let (no,n') ← expandNested functions constants used no n
        next:=n'
        `(tritonStmt| if $condition { $yes* } else { $no* })
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* })=>do
        noEscapes (functions.map (·.name)) condition
        let (yes,n) ← expandNested functions constants used yes next
        next:=n
        `(tritonStmt| if $condition { $yes* })
      | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $step:tritonExpr) { $nested:tritonStmt* })=>do
        for bound in #[lo,hi,step] do noEscapes (functions.map (·.name)) bound
        let (nested,n) ← expandNested functions constants used nested next
        next:=n
        `(tritonStmt| for $counter:ident in range($lo, $hi, $step) { $nested* })
      | _=>do
        noEscapes (functions.map (·.name)) stmt
        pure stmt
    lowered:=lowered.push stmt
  return (lowered,next)

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
  let mut fixed : List (Name × Nat) := []
  for name in constants do
    let value ← elabTerm (mkIdent name) none
    if ← isDefEq (← inferType value) (mkConst ``Nat) then
      if let .lit (.natVal n) ← whnf value then fixed:=fixed++[(name,n)]
  for stmt in body do rejectWrites (fixed.map Prod.fst) stmt.raw
  let body := (← liftMacroM (select fixed body.toList)).toArray
  for stmt in body do immutableArguments constants stmt
  let used:=occupied++ids stx++body.toList.flatMap (ids ∘ TSyntax.raw)++
    functions.flatMap (fun f=>f.parameters++f.locals)
  let (expanded,_) ← expandNested functions constants used body 0
  let mut lowered:=expanded
  let queries:List Name:=(lowered.toList.flatMap (elementQueries ∘ TSyntax.raw)).eraseDups
  let mut elements:List (Name × Syntax):=[]
  for pointer in queries do
    for stmt in body do checkPointerRebindings pointer stmt
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
  let mut booleans : List (Name × Syntax):=[]
  for name in constants do
    let value ← elabTerm (mkIdent name) none
    if ← isDefEq (← inferType value) (mkConst ``Bool) then
      let term : TSyntax `term:=⟨mkIdent name⟩
      let expression ← liftMacroM `(tritonExpr| $(($term:term : Bool)))
      booleans:=booleans++[(name,expression.raw)]
  let names ← mkListLit (mkConst ``String) (booleans.map (mkStrLit ∘ Name.toString ∘ Prod.fst))
  let selected ← exprToSyntax (← mkAppM ``withoutBooleanNames #[profile,names])
  lowered:=lowered.map fun statement=>(⟨bindBooleans booleans statement.raw⟩ : TSyntax `tritonStmt)
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (lowered.toList.map (renderPython ·.raw))++"\n}")
  let mut callBindings : Array (TSyntax `tritonCallBinding):=#[]
  let mut abiArguments : Array (TSyntax `term):=#[]
  let scalarTypes ← listValues (← mkAppM ``CallSignature.scalars #[profile])
  for binding in stx[4].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding | throwError "named call argument required"
    let mut isFP32:=Bool.false
    for pair in scalarTypes do
      let key ← stringValue (← mkAppM ``Prod.fst #[pair])
      if key==name.getId.eraseMacroScopes.toString then
        let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
        isFP32:=dtype.isConstOf ``ScalarType.fp32
    if isFP32 then
      let _ ← elabTermEnsuringType value (mkConst ``Real)
      let label:=quote name.getId.eraseMacroScopes.toString
      abiArguments:=abiArguments.push (← `(ScalarArgument.fp32 $label $value))
    else callBindings:=callBindings.push ⟨binding⟩
  let base ← `(with_triton_call $selected:term call_args [$callBindings,*] in triton {})
  let base:=base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)
  let kernel ← SignedAugmentedAssignments.elaborateInvocation base (some (mkConst ``ComputeKernel))
  if abiArguments.isEmpty then return ← ensureHasType expected kernel
  let args ← elabTermEnsuringType (← `([$abiArguments,*])) (mkApp (mkConst ``List [levelZero]) (mkConst ``ScalarArgument))
  ensureHasType expected (← mkAppM ``ComputeKernel.withScalarArguments #[kernel,args])


end VeriTile.Triton.DSL.RecursiveJitFunctions
