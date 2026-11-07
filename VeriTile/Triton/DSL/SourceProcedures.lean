import VeriTile.Triton.DSL.ProcedureJitFunctions
import VeriTile.Triton.DSL.SymbolicFloatConstants

/-! Storage-polymorphic pasted source procedures. Each call is elaborated in
its own parameter scope with storage inferred from the caller ABI. Only the
resulting ordinary statements are concatenated; no numerical or result port
is introduced. The same source can therefore receive FP16 or byte pointers.
Caller source currently consists of complete procedure calls. -/
namespace VeriTile.Triton
structure SourceProcedure where
  parameters : List String
  constexpr : List String
  materialized : List String
  source : String

namespace DSL.SourceProcedures
open Lean Elab Term Meta
syntax (name := sourceProcedure) "triton_source_procedure " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" " fp32_constants " "[" ident,* "]"
  RawSourceText.body : term
/-- Declare the arguments of a reusable pasted Triton body. -/
syntax (name := sourceParams) "with_triton_params " "(" ident,* ")"
  " constexpr_args " "[" ident,* "]" (" fp32_constants " "[" ident,* "]")?
  " in " "triton " RawSourceText.body : term
syntax (name := sourceInstance) "instantiate_source_procedure " term
  " signature " term " procedure_args " "[" tritonCallBinding,* "]" : term
syntax (name := sourceCalls) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" (" fp32_constants " "[" ident,* "]")?
  " source_procedures " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private def simple (name : Name) : Bool :=
  name.isStr && name.getPrefix.isAnonymous &&
    !["tl","triton","float","True","False","None"].contains name.toString
private partial def tokens : Syntax → List String
  | .atom _ text=>[text]
  | .ident _ spelling name _=>
    let name:=name.eraseMacroScopes.toString
    [if name=="do" || name=="_" then "«"++name++"»" else spelling.toString]
  | .node _ kind args=>if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing=>[]
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out
private partial def first (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then first raw[0]! else raw.setArgs (raw.getArgs.map first)
private partial def normalizeRanges (raw : Syntax) : MacroM Syntax := do
  let raw:=first raw
  let raw:=raw.setArgs (← raw.getArgs.mapM normalizeRanges)
  match raw with
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident))=>
    let target:TSyntax `term:=⟨target.raw⟩
    return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($target:term))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident))=>
    let target:TSyntax `term:=⟨target.raw⟩
    return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($target:term))).raw
  | `(tritonStmt| for $counter:ident in range($start:tritonExpr, $stop:tritonExpr, -$stride:tritonExpr) { $body:tritonStmt* })=>
    return (← `(tritonStmt| for $counter:ident in range($start:tritonExpr, -(0 - ($stop:tritonExpr)), -$stride:tritonExpr) { $body* })).raw
  | `(tritonStmt| for $counter:ident in range(0, $stop:tritonExpr, $stride:tritonExpr) { $body:tritonStmt* })=>
    return (← `(tritonStmt| for $counter:ident in range((0), $stop:tritonExpr, $stride:tritonExpr) { $body* })).raw
  | `(tritonStmt| for $counter:ident in range(0, $stop:tritonExpr) { $body:tritonStmt* })=>
    return (← `(tritonStmt| for $counter:ident in range((0), $stop:tritonExpr, 1) { $body* })).raw
  | `(tritonStmt| for $counter:ident in range($stop:tritonExpr) { $body:tritonStmt* })=>
    return (← `(tritonStmt| for $counter:ident in range((0), $stop:tritonExpr, 1) { $body* })).raw
  | _=>return raw
private def prepare (source : String) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let base ← `(with_triton_call ({} : CallSignature) call_args [] in triton {})
  let raw:=Syntax.atom .none source
  let prepared ← PythonTensorTuples.prepare (base.raw.setArgs
    (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw))
  let raw:=prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body=>pure body | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete procedure body required"
  return ← body.mapM (fun statement=>return ⟨← liftMacroM (normalizeRanges statement.raw)⟩)
private def emptyReturn (statement : TSyntax `tritonStmt) : Bool :=
  match statement with
  | `(tritonStmt| return $values:tritonExpr,*)=>values.getElems.isEmpty
  | _=>Bool.false

private partial def checkExpr (bound : List Name) (raw : Syntax) : TermElabM Unit := do
  if raw.isAtom && raw.getAtomVal.contains '$' then throwErrorAt raw "Lean expressions cannot occur in a source procedure"
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    let root:=(name.toString.splitOn ".").head!.toName
    unless bound.contains root || name.toString.startsWith "tl." ||
        ["float","True","False","None"].contains name.toString do
      throwErrorAt raw "procedure references an unbound name: {name}"
  let args:=raw.getArgs
  for i in [:args.size] do
    let key:=args[i]!.isIdent && args[i+1]?.any
      (fun next=>next.isAtom && next.getAtomVal.trimAscii.toString=="=")
    unless key do checkExpr bound args[i]!

private partial def validateStatements (body : Array (TSyntax `tritonStmt))
    (bound parameters : List Name) : TermElabM (List Name × List Name) := do
  let mut bound:=bound
  let mut locals:List Name:=[]
  for statement in body do
    match statement with
    | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr, $stride:tritonExpr) { $nested:tritonStmt* })=>
      for e in [lo,hi,stride] do checkExpr bound e
      let name:=counter.getId.eraseMacroScopes
      unless simple name && !parameters.contains name do throwErrorAt counter "procedure loop cannot rebind a parameter"
      let (_,nestedLocals) ← validateStatements nested (name::bound) parameters
      locals:=(locals++[name]++nestedLocals).eraseDups
    | `(tritonStmt| for $counter:ident in range($lo:tritonExpr, $hi:tritonExpr) { $nested:tritonStmt* })=>
      for e in [lo,hi] do checkExpr bound e
      let name:=counter.getId.eraseMacroScopes
      unless simple name && !parameters.contains name do throwErrorAt counter "procedure loop cannot rebind a parameter"
      let (_,nestedLocals) ← validateStatements nested (name::bound) parameters
      locals:=(locals++[name]++nestedLocals).eraseDups
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>
      checkExpr bound condition
      let (yb,yl) ← validateStatements yes bound parameters
      let (nb,nl) ← validateStatements no bound parameters
      bound:=yb.filter nb.contains
      locals:=(locals++yl++nl).eraseDups
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* })=>
      checkExpr bound condition
      let (_,yl) ← validateStatements yes bound parameters
      locals:=(locals++yl).eraseDups
    | _=>
      if emptyReturn statement then throwErrorAt statement "only a final empty procedure return is supported"
      let args:=statement.raw.getArgs
      let assignment:=args.findIdx? (fun arg=>arg.isAtom &&
        ["=","+=","-=","*=","/="].contains arg.getAtomVal.trimAscii.toString)
      match assignment with
      | some index=>
        let names:=(args.extract 0 index).toList.flatMap ids
        unless !names.isEmpty && names.all simple && !(names.any parameters.contains) do
          throwErrorAt statement "source constexpr parameters remain immutable"
        let augmented:=args[index]!.getAtomVal.trimAscii.toString != "="
        if augmented && !(names.all bound.contains) then throwErrorAt statement "augmented assignment needs an existing local"
        for arg in args.extract (index+1) args.size do checkExpr bound arg
        locals:=(locals++names).eraseDups
        bound:=(bound++names).eraseDups
      | none=>checkExpr bound statement
  return (bound,locals)

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  unless value.isAppOfArity ``List.cons 3 do throwError "static procedure metadata required"
  return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "static procedure strings required"
  return text
private def names (value : Expr) : TermElabM (List Name) := do
  (← listValues value).mapM (fun value=>return (← textValue value).toName)
private def profileNames (profile : Expr) : TermElabM (List Name) := do
  let mut result:List Name:=[]
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for entry in ← listValues (← mkAppM field #[profile]) do
      result:=result++[(← textValue (← mkAppM ``Prod.fst #[entry])).toName]
  return result++(← names (← mkAppM ``CallSignature.constexpr #[profile]))

private structure Function where
  parameters : List Name
  constants : List Name
  materialized : List Name
  source : String

private def validate (f : Function) : TermElabM Unit := do
  unless !f.parameters.isEmpty && f.parameters.all simple &&
      f.parameters.eraseDups==f.parameters && f.constants.eraseDups==f.constants &&
      f.constants.all f.parameters.contains && f.materialized.eraseDups==f.materialized &&
      f.materialized.all f.constants.contains do
    throwError "distinct source parameters and constexpr subsets required"
  let body ← prepare f.source
  discard <| validateStatements body f.parameters f.constants

private def function (value : Expr) : TermElabM Function := do
  unless (← inferType value).isConstOf ``SourceProcedure do throwError "pasted source procedure required"
  let f : Function := ⟨← names (← mkAppM ``SourceProcedure.parameters #[value]),
    ← names (← mkAppM ``SourceProcedure.constexpr #[value]),
    ← names (← mkAppM ``SourceProcedure.materialized #[value]),
    ← textValue (← mkAppM ``SourceProcedure.source #[value])⟩
  validate f
  return f

private def elaborateProcedure (stx materialized : Syntax) (expected : Option Expr) :
    TermElabM Expr := do
  let f : Function := ⟨stx[2].getSepArgs.toList.map (·.getId.eraseMacroScopes),
    stx[6].getSepArgs.toList.map (·.getId.eraseMacroScopes),
    materialized.getSepArgs.toList.map (·.getId.eraseMacroScopes),stx.getArgs.back!.getAtomVal⟩
  validate f
  let strings (values : List Name):=mkListLit (mkConst ``String) (values.map (mkStrLit ∘ Name.toString))
  ensureHasType expected (mkApp4 (mkConst ``SourceProcedure.mk)
    (← strings f.parameters) (← strings f.constants) (← strings f.materialized) (mkStrLit f.source))

@[term_elab sourceProcedure]
def elaborateSource : TermElab := fun stx expected =>
  elaborateProcedure stx stx[10] expected

@[term_elab sourceParams]
def elaborateParams : TermElab := fun stx expected =>
  elaborateProcedure stx stx[8][2] expected

private def instantiate (f : Function) (profile : Expr) (values : List (Name × Expr))
    (expected : Option Expr) (literalBooleans : List Name := []) : TermElabM Expr := do
  let declared ← profileNames profile
  unless declared.length==f.parameters.length && declared.all f.parameters.contains &&
      (← names (← mkAppM ``CallSignature.constexpr #[profile]))==f.constants do
    throwError "procedure ABI must cover exactly its original parameters and constexprs"
  let required:=f.parameters.filter (!f.materialized.contains ·)
  unless (values.map (·.1)).eraseDups.length == required.length &&
      values.length==required.length && (values.all fun pair=>required.contains pair.1) do
    throwError "each non-materialized source parameter must be supplied exactly once"
  let rec bind (todo : List (Name × Expr)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail=>do
        let dtype ← inferType value
        if literalBooleans.contains name then
          unless dtype.isConstOf ``Bool do throwError "literal Boolean argument must have Bool type"
          -- Keep the source branch in the IR. Substitute its closed value only
          -- after elaboration, so a call and an isolated callee have the same
          -- ordinary constant-condition statement, with identical semantics.
          withLocalDeclD name dtype fun binding=>do
            return (← bind tail).replaceFVar binding value
        else
          withLetDecl name dtype value fun binding=>do
            mkLetFVars #[binding] (← bind tail)
    | []=>
      let signatureSyntax ← exprToSyntax profile
      let mut args:Array (TSyntax `tritonCallBinding):=#[]
      for scalar in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
        let name:=mkIdent (← textValue (← mkAppM ``Prod.fst #[scalar])).toName
        args:=args.push (← `(tritonCallBinding| $name:ident := $name:ident))
      let constants:=f.materialized.toArray.map mkIdent
      let body:=Syntax.atom .none f.source
      if constants.isEmpty then
        let base ← `(with_triton_call $signatureSyntax:term call_args [$args,*] in triton {})
        PythonTensorTuples.elaborateInvocation
          (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) body)) expected
      else
        let base ← `(with_triton_call $signatureSyntax:term call_args [$args,*]
          fp32_constants [$constants,*] in triton {})
        SymbolicFloatConstants.elaborateInvocation
          (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) body)) expected
  bind values

@[term_elab sourceInstance]
def elaborateInstance : TermElab := fun stx expected=>do
  let f ← function (← elabTerm stx[1] none)
  let profile ← elabTermEnsuringType stx[3] (mkConst ``CallSignature)
  let mut values:List (Name × Expr):=[]
  for binding in stx[6].getSepArgs do
    let `(tritonCallBinding| $name:ident := $term:term):=binding | throwError "named arguments required"
    values:=values++[(name.getId.eraseMacroScopes,← elabTerm term none)]
  instantiate f profile values expected

private def signatureFor (caller : Expr) (f : Function) (actuals : List (Name × Name)) : TermElabM Expr := do
  let field (projection type : Name) : TermElabM Expr := do
    let mut entries:List Expr:=[]
    for (formal,actual) in actuals do
      if f.constants.contains formal then continue
      for supplied in ← listValues (← mkAppM projection #[caller]) do
        if (← textValue (← mkAppM ``Prod.fst #[supplied])).toName==actual then
          entries:=entries++[← mkAppM ``Prod.mk #[mkStrLit formal.toString,
            ← mkAppM ``Prod.snd #[supplied]]]
    mkListLit (mkApp2 (mkConst ``Prod [levelZero,levelZero]) (mkConst ``String) (mkConst type)) entries
  let pointers ← field ``CallSignature.pointers ``SourceDType
  let floats ← field ``CallSignature.floatingPointers ``FloatDType
  let scalars ← field ``CallSignature.scalars ``ScalarType
  let constants ← mkListLit (mkConst ``String) (f.constants.map (mkStrLit ∘ Name.toString))
  mkAppM ``CallSignature.mk #[pointers,floats,scalars,constants,
    ← mkListLit (mkApp2 (mkConst ``Prod [levelZero,levelZero]) (mkConst ``String) (mkConst ``String)) [],
    ← mkListLit (mkApp2 (mkConst ``Prod [levelZero,levelZero]) (mkConst ``String) (mkConst ``String)) [],
    ← mkListLit (mkConst ``String) [],← mkListLit (mkConst ``String) []]

private def kernelInputs : ComputeKernel → List RegionName
  | .mk inputs _ _=>inputs
private def kernelOutputs : ComputeKernel → List RegionName
  | .mk _ outputs _=>outputs

@[term_elab sourceCalls]
def elaborateCalls : TermElab := fun stx expected=>do
  let caller ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let initial ← profileNames caller
  unless initial.eraseDups==initial do throwError "caller ABI names must be distinct"
  let constants ← names (← mkAppM ``CallSignature.constexpr #[caller])
  let materialized:=stx[6][2].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  unless materialized.eraseDups==materialized && materialized.all constants.contains do
    throwError "caller materialized constants must be distinct constexprs"
  let mut bindings:List (Name × Expr):=[]
  for binding in stx[4].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term):=binding | throwError "named caller scalars required"
    bindings:=bindings++[(name.getId.eraseMacroScopes,← elabTerm value none)]
  let scalarNames ← (← listValues (← mkAppM ``CallSignature.scalars #[caller])).mapM
    (fun entry=>return (← textValue (← mkAppM ``Prod.fst #[entry])).toName)
  unless (bindings.map (·.1)).eraseDups.length==bindings.length &&
      bindings.length==scalarNames.length && (bindings.all fun pair=>scalarNames.contains pair.1) do
    throwError "caller runtime scalars require exact distinct bindings"
  let mut functions:List (Name × Function):=[]
  for binding in stx[9].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term):=binding | throwError "named source procedures required"
    let name:=name.getId.eraseMacroScopes
    unless simple name && !(functions.any (·.1==name)) do throwError "distinct procedure names required"
    functions:=functions++[(name,← function (← elabTerm value none))]
  let mut kernels:List Expr:=[]
  for statement in ← prepare stx.getArgs.back!.getAtomVal do
    let `(tritonStmt| $callee:ident($arguments:statementJitArgument,*)):=statement |
      throwErrorAt statement "source procedure caller requires complete top-level calls"
    let some (_,f):=functions.find? (·.1==callee.getId.eraseMacroScopes) |
      throwErrorAt callee "unknown source procedure"
    let mut literalBooleans:List Name:=[]
    let mut actuals:List (Name × Name):=[]
    let mut values:List (Name × Expr):=[]
    let mut positional:=0
    let mut keywordSeen:=Bool.false
    for argument in arguments.getElems do
      let (formal,value) ← match argument with
        | `(statementJitArgument| $key:ident = $value:tritonExpr)=>
          keywordSeen:=Bool.true; pure (key.getId.eraseMacroScopes,value)
        | `(statementJitArgument| $value:tritonExpr)=>do
          if keywordSeen then throwErrorAt argument "positional argument follows keyword"
          let some parameter:=f.parameters[positional]? | throwError "too many source arguments"
          positional:=positional+1; pure (parameter,value)
        | _=>throwError "unsupported source argument"
      unless f.parameters.contains formal && !(actuals.any (·.1==formal)) do
        throwErrorAt argument "unexpected or repeated source argument"
      let text:=render value.raw
      let literal:=text=="True" || text=="False" || text=="true" || text=="false"
      if literal then
        unless f.constants.contains formal && !f.materialized.contains formal do
          throwErrorAt argument "Boolean literal requires a non-materialized constexpr"
        literalBooleans:=literalBooleans++[formal]
        actuals:=actuals++[(formal,formal)]
        values:=values++[(formal,Lean.mkConst (if text=="True" || text=="true" then ``Bool.true else ``Bool.false))]
      else
        let `(tritonExpr| $name:ident):=value | throwErrorAt value "source call arguments must be stable caller names"
        let actual:=name.getId.eraseMacroScopes
        unless initial.contains actual do throwErrorAt value "argument not declared in caller ABI"
        if f.constants.contains formal && !constants.contains actual then
          throwErrorAt value "constexpr requires caller constexpr"
        actuals:=actuals++[(formal,actual)]
        if f.materialized.contains formal then
          unless formal==actual && materialized.contains actual do
            throwErrorAt value "materialized scalar must retain its declared register name"
        else
          let expression ← match bindings.find? (·.1==actual) with
            | some (_,e)=>pure e
            | none=>elabTerm (mkIdent actual) none
          values:=values++[(formal,expression)]
    unless actuals.length==f.parameters.length do throwError "every source parameter must be provided"
    let profile ← signatureFor caller f actuals
    kernels:=kernels++[← instantiate f profile values (some (mkConst ``ComputeKernel)) literalBooleans]
  let ports (projection : Name) (fallback : Name) : TermElabM Expr := do
    let declared ← names (← mkAppM projection #[caller])
    if !declared.isEmpty then
      unless declared.eraseDups==declared && declared.all initial.contains do
        throwError "explicit procedure ports must be distinct caller arguments"
      let values ← declared.mapM fun name=>elabTerm (mkIdent name) (some (mkConst ``RegionName))
      return ← mkListLit (mkConst ``RegionName) values
    let mut result ← mkListLit (mkConst ``RegionName) []
    for kernel in kernels do result ← mkAppM ``List.append #[result,← mkAppM fallback #[kernel]]
    return result
  let inputs ← ports ``CallSignature.readPorts ``kernelInputs
  let outputs ← ports ``CallSignature.writePorts ``kernelOutputs
  let stages ← mkListLit (mkConst ``ComputeKernel) kernels
  ensureHasType expected (← mkAppM ``ComputeKernel.seq #[inputs,outputs,stages])

end DSL.SourceProcedures
end VeriTile.Triton
