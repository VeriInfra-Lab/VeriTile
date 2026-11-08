import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.FiniteNatJitFunctions
import VeriTile.Triton.DSL.SignedCallRanges
import VeriTile.Triton.DSL.StraightLineIdentifiers

/-! Complete source procedures with memory effects and implicit/empty return.
Parameters stay immutable. Calls accept only stable named arguments. Callee
locals cannot collide with caller bindings or escape into later caller code;
independent calls may reuse their private scratch names. This permits ordinary
Triton helper calls without pasting their bodies into the caller quotation. -/
namespace VeriTile.Triton
structure ProcedureJitFunction where
  signature : CallSignature
  parameters : List String
  constexpr : List String
  source : String

namespace DSL.ProcedureJitFunctions
open Lean Elab Term Meta
syntax (name := procedureBody) "triton_procedure " term:max " (" ident,* ")"
  " constexpr_args " "[" ident,* "]" RawSourceText.body : term
syntax (name := instantiateProcedure) "instantiate_triton_procedure " term
  " procedure_args " "[" tritonCallBinding,* "]" : term
syntax (name := withProcedures) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " jit_procedures " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term
syntax (name := effectCall) (priority := low) ident "(" statementJitArgument,* ")" : tritonStmt

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
  let prepared ← PythonFullConstructors.prepareSource (base.raw.setArgs
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
          throwErrorAt statement "procedure parameters remain immutable"
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
/-- A call cannot silently reinterpret an argument's storage or scalar kind.
Computed integer launch coordinates may be caller locals; floating scalars and
pointers must come from an explicitly typed caller argument. -/
private def checkActual (callee caller : Expr) (parameter actual : Name) : TermElabM Unit := do
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for entry in ← listValues (← mkAppM field #[callee]) do
      if (← textValue (← mkAppM ``Prod.fst #[entry])).toName==parameter then
        let dtype ← mkAppM ``Prod.snd #[entry]
        for supplied in ← listValues (← mkAppM field #[caller]) do
          if (← textValue (← mkAppM ``Prod.fst #[supplied])).toName==actual then
            unless ← isDefEq dtype (← mkAppM ``Prod.snd #[supplied]) do
              throwError "procedure argument type differs from its declared ABI"
            return
        if field==``CallSignature.scalars && (← whnf dtype).isConstOf ``ScalarType.integer &&
            !((← profileNames caller).contains actual) then return
        throwError "procedure argument lacks a matching caller ABI entry"

private def sourceBody (source : String) (parameters : List Name) :
    TermElabM (Array (TSyntax `tritonStmt) × List Name) := do
  let body ← prepare source
  let body:=if body.back?.any emptyReturn then body.pop else body
  let (_,locals) ← validateStatements body parameters parameters
  return (body,locals)

@[term_elab procedureBody]
def elaborateProcedure : TermElab := fun stx expected=>do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let parameters:=stx[3].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  let constants:=stx[7].getSepArgs.toList.map (·.getId.eraseMacroScopes)
  unless !parameters.isEmpty && parameters.all simple && parameters.eraseDups==parameters &&
      constants.eraseDups==constants && constants.all parameters.contains do throwError "distinct source procedure parameters required"
  let declared ← profileNames profile
  unless declared.length==parameters.length && declared.all parameters.contains do throwError "procedure ABI does not cover its exact parameters"
  unless (← names (← mkAppM ``CallSignature.constexpr #[profile]))==constants do throwError "procedure constexpr declaration differs from its ABI"
  let raw:=stx.getArgs.back!
  let source:=raw.getAtomVal
  discard <| sourceBody source parameters
  let strings (values : List Name):=mkListLit (mkConst ``String) (values.map (mkStrLit ∘ Name.toString))
  ensureHasType expected (mkApp4 (mkConst ``ProcedureJitFunction.mk) profile
    (← strings parameters) (← strings constants) (mkStrLit source))

private structure Function where
  profile : Expr
  parameters : List Name
  constants : List Name
  body : Array (TSyntax `tritonStmt)
  locals : List Name
private def function (value : Expr) : TermElabM Function := do
  unless (← inferType value).isConstOf ``ProcedureJitFunction do throwError "pasted procedure required"
  let profile ← mkAppM ``ProcedureJitFunction.signature #[value]
  let parameters ← names (← mkAppM ``ProcedureJitFunction.parameters #[value])
  let constants ← names (← mkAppM ``ProcedureJitFunction.constexpr #[value])
  let (body,locals) ← sourceBody (← textValue (← mkAppM ``ProcedureJitFunction.source #[value])) parameters
  return ⟨profile,parameters,constants,body,locals⟩

/-- Capture both tuple values before either source destination is changed.
Fresh names avoid every identifier in the complete caller. -/
private partial def captureTuples (used : List Name) (next : Nat) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt) × Nat) := do
  let mut out:=#[]
  let mut next:=next
  for statement in body do
    let raw:=statement.raw
    if raw.getNumArgs==7 && raw[0]!.isIdent && raw[2]!.isIdent &&
        raw[1]!.getAtomVal.trimAscii.toString=="," && raw[3]!.getAtomVal.trimAscii.toString=="=" then
      let a:Ident:=⟨raw[0]!⟩
      let b:Ident:=⟨raw[2]!⟩
      let x:TSyntax `tritonExpr:=⟨raw[4]!⟩
      let y:TSyntax `tritonExpr:=⟨raw[6]!⟩
      let fresh (start : Nat) : Ident × Nat:=Id.run do
        let mut n:=start
        let mut name:=("__veritile_procedure_tuple_"++toString n).toName
        while used.contains name do
          n:=n+1
          name:=("__veritile_procedure_tuple_"++toString n).toName
        return (mkIdent name,n+1)
      let (left,n):=fresh next
      let (right,n):=fresh n
      next:=n
      out:=out++#[← `(tritonStmt| $left:ident = $x:tritonExpr),
        ← `(tritonStmt| $right:ident = $y:tritonExpr),
        ← `(tritonStmt| $a:ident = $left:ident),← `(tritonStmt| $b:ident = $right:ident)]
      continue
    match statement with
    | `(tritonStmt| for $i:ident in range($lo:tritonExpr, $hi:tritonExpr, $step:tritonExpr) { $inner:tritonStmt* })=>
      let (inner,n) ← captureTuples used next inner
      next:=n
      out:=out.push (← `(tritonStmt| for $i:ident in range($lo:tritonExpr, $hi:tritonExpr, $step:tritonExpr) { $inner* }))
    | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* })=>
      let (yes,n) ← captureTuples used next yes
      let (no,n) ← captureTuples used n no
      next:=n
      out:=out.push (← `(tritonStmt| if $cond:tritonExpr { $yes* } else { $no* }))
    | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* })=>
      let (yes,n) ← captureTuples used next yes
      next:=n
      out:=out.push (← `(tritonStmt| if $cond:tritonExpr { $yes* }))
    | _=>out:=out.push statement
  return (out,next)

@[term_elab instantiateProcedure]
def elaborateInstance : TermElab := fun stx expected=>do
  let f ← function (← elabTerm stx[1] none)
  let mut values:List (Name × Expr):=[]
  for binding in stx[4].getSepArgs do
    let `(tritonCallBinding| $name:ident := $term:term):=binding | throwError "named procedure arguments required"
    let name:=name.getId.eraseMacroScopes
    unless f.parameters.contains name && !(values.any (·.1==name)) do throwError "unexpected or repeated procedure argument"
    values:=values++[(name,← elabTerm term none)]
  unless values.length==f.parameters.length do throwError "every procedure argument must be bound"
  let rec bind (todo : List (Name × Expr)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail=>withLetDecl name (← inferType value) value fun binding=>do
        mkLetFVars #[binding] (← bind tail)
    | []=>
      let profile ← exprToSyntax f.profile
      let mut arguments:Array (TSyntax `tritonCallBinding):=#[]
      for scalar in ← listValues (← mkAppM ``CallSignature.scalars #[f.profile]) do
        let name:=mkIdent (← textValue (← mkAppM ``Prod.fst #[scalar])).toName
        arguments:=arguments.push (← `(tritonCallBinding| $name:ident := $name:ident))
      let base ← `(with_triton_call $profile:term call_args [$arguments,*] in triton {})
      let (body,_) ← liftMacroM (captureTuples (f.body.toList.flatMap (ids ∘ TSyntax.raw)) 0 f.body)
      let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (body.toList.map (render ∘ TSyntax.raw))++"\n}")
      elabTerm (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)) expected
  bind values

private partial def substitute (arguments : List (Name × Name)) (raw : Syntax) : Syntax :=
  if raw.isIdent then
    let name:=raw.getId.eraseMacroScopes
    match arguments.find? (fun pair=>name==pair.1 || name.toString.startsWith (pair.1.toString++".")) with
    | some (parameter,actual)=>
      let suffix:=(name.toString.drop parameter.toString.length).toString
      mkIdent (actual.toString++suffix).toName
    | none=>raw
  else
    let args:=raw.getArgs
    raw.setArgs (args.mapIdx fun i arg=>
      if arg.isIdent && args[i+1]?.any (fun next=>next.isAtom && next.getAtomVal.trimAscii.toString=="=")
      then arg else substitute arguments arg)

@[term_elab withProcedures]
def elaborateCalls : TermElab := fun stx expected=>do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut functions:List (Name × Function):=[]
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term):=binding | throwError "named procedures required"
    let name:=name.getId.eraseMacroScopes
    unless simple name && !(functions.any (·.1==name)) do throwError "distinct procedure names required"
    functions:=functions++[(name,← function (← elabTerm value none))]
  let initial ← profileNames profile
  let constants ← names (← mkAppM ``CallSignature.constexpr #[profile])
  let mut bound:=initial
  let body ← prepare stx.getArgs.back!.getAtomVal
  let mut out:Array (TSyntax `tritonStmt):=#[]
  for statement in body do
    if let `(tritonStmt| $callee:ident($arguments:statementJitArgument,*)):=statement then
      if let some (_,f):=functions.find? (·.1==callee.getId.eraseMacroScopes) then
        if f.locals.any bound.contains then throwErrorAt statement "procedure local would capture a caller binding"
        let mut actuals:List (Name × Name):=[]
        let mut positional:=0
        let mut keywordSeen:=Bool.false
        for argument in arguments.getElems do
          let (parameter,value) ← match argument with
            | `(statementJitArgument| $key:ident = $value:tritonExpr)=>
              keywordSeen:=Bool.true
              pure (key.getId.eraseMacroScopes,value)
            | `(statementJitArgument| $value:tritonExpr)=>do
              if keywordSeen then throwErrorAt argument "positional argument follows a keyword"
              let some parameter:=f.parameters[positional]? | throwError "too many procedure arguments"
              positional:=positional+1
              pure (parameter,value)
            | _=>throwError "unsupported procedure argument"
          unless f.parameters.contains parameter && !(actuals.any (·.1==parameter)) do throwError "unexpected or repeated procedure argument"
          let `(tritonExpr| $actual:ident):=value | throwErrorAt value "procedure calls require stable named arguments"
          let actual:=actual.getId.eraseMacroScopes
          unless bound.contains actual do throwErrorAt value "procedure argument is not bound by the caller"
          if f.constants.contains parameter && !constants.contains actual then
            throwErrorAt value "procedure constexpr needs a caller constexpr"
          checkActual f.profile profile parameter actual
          actuals:=actuals++[(parameter,actual)]
        unless actuals.length==f.parameters.length do throwError "every procedure argument must be provided"
        for nested in f.body do out:=out.push ⟨substitute actuals nested.raw⟩
        continue
    if (ids statement).any (fun name=>functions.any (·.1==name)) then
      throwErrorAt statement "a procedure can only occur as a complete top-level call"
    let (next,_) ← validateStatements #[statement] bound initial
    bound:=next
    out:=out.push statement
  let signature:TSyntax `term:=⟨stx[1]⟩
  let arguments:=stx[4].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let base ← `(with_triton_call $signature:term call_args [$arguments,*] in triton {})
  let (lowered,_) ← liftMacroM (captureTuples (out.toList.flatMap (ids ∘ TSyntax.raw)) 0 out)
  let raw:=Syntax.atom stx.getHeadInfo ("{\n"++String.intercalate "\n" (lowered.toList.map (render ∘ TSyntax.raw))++"\n}")
  elabTerm (base.raw.setArgs (base.raw.getArgs.set! (base.raw.getArgs.size-1) raw)) expected
end DSL.ProcedureJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.ProcedureJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceSyntax.typedParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``ProcedureJitFunction do throwUnsupportedSyntax
  let parameters := stx[2].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let constants := stx[6].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let profileSyntax : TSyntax `term := ⟨stx[9]⟩
  let body ← `(triton_procedure $profileSyntax:term ($parameters,*) constexpr_args [$constants,*] {})
  elaborateProcedure (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.ProcedureJitFunctions
