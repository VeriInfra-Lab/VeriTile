import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.RandomProfile
import VeriTile.Triton.DSL.UniformJitFunctions

/-! Invocation bindings for word-preserving random kernels. FP32 arguments are
bound by their ABI words, signed arguments by their explicit Int64 channel,
and symbolic arange sizes by a checked identity adapter. A reviewed module
helper may name only the canonical uint-to-uniform decoder. These declarations
precede the copied body; none replaces a source instruction or random call. -/
namespace VeriTile.Triton.DSL.PhiloxCallArguments
open Lean Elab Term Meta

def withoutArguments (profile : CallSignature) (integers sizes : List String) : CallSignature :=
  { profile with scalars := profile.scalars.filter (fun p => !integers.contains p.1)
                 constexpr := profile.constexpr.filter (fun p => !sizes.contains p) }

syntax (name := withArguments) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " fp32_bits " "[" tritonCallBinding,* "]"
  " i64_args " "[" tritonCallBinding,* "]" " arange_args " "[" tritonCallBinding,* "]"
  " uniform_helpers " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "random call profiles must be statically known"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (immutable : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for op in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==op) then
      if ((args.take i).flatMap ids).any immutable.contains then
        throwErrorAt raw "word arguments, arange sizes and uniform helpers must remain immutable: {raw}"
  if let first::index::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids index).any immutable.contains then
      throwError "a loop cannot shadow a random call binding"
  for child in raw.getArgs do rejectWrites immutable child

private partial def checkIntegerWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  if let `(tritonStmt| $out:ident = $value:tritonExpr) := raw then
    if names.contains out.getId.eraseMacroScopes then
      if let some (input,dtype) := methodCast? value then
        let inputName := if input.raw.isIdent then some input.raw.getId.eraseMacroScopes else
          match input with | `(tritonExpr| $id:ident) => some id.getId.eraseMacroScopes | _ => none
        if inputName==some out.getId.eraseMacroScopes &&
            (match dtype with | `(tritonDType| tl.int64) => Bool.true | _ => Bool.false) then return
  rejectWrites names raw

/-- Escape only a source identifier token reserved by Lean, retaining strings,
comments and the name of the emitted register. -/
private def escapeTo (source : String) : String := Id.run do
  let mut result := ""
  let mut word := ""
  let mut quote : Option Char := none
  let mut escaped := Bool.false
  let mut comment := Bool.false
  for c in source.toList do
    if comment then
      result:=result.push c
      if c=='\n' then comment:=Bool.false
    else if let some delimiter := quote then
      result:=result.push c
      if escaped then escaped:=Bool.false
      else if c=='\\' then escaped:=Bool.true
      else if c==delimiter then quote:=none
    else if c.isAlphanum || c=='_' then word:=word.push c
    else
      result:=result++(if word=="to" && !result.endsWith "." then "«to»" else word)
      word:=""
      result:=result.push c
      if c=='"' || c=='\'' then quote:=some c
      if c=='#' then comment:=Bool.true
  return result++(if word=="to" && !result.endsWith "." then "«to»" else word)

@[term_elab withArguments] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let scalars ← listValues (← mkAppM ``CallSignature.scalars #[profile])
  let mut scalarKinds : List (Name × Expr) := []
  for pair in scalars do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair]) | throwError "literal scalar names required"
    scalarKinds:=scalarKinds++[(name.toName,← whnf (← mkAppM ``Prod.snd #[pair]))]
  let mut constexpr : List Name := []
  for value in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf value | throwError "literal constexpr names required"
    constexpr:=constexpr++[name.toName]
  let read (raw : Syntax) (kind type : Name) : TermElabM (List (Name × Expr)) := do
    let mut result := []
    for entry in raw.getSepArgs do
      let `(tritonCallBinding| $name:ident := $value:term) := entry | throwError "named random argument required"
      let name:=name.getId.eraseMacroScopes
      unless scalarKinds.any (fun p => p.1==name && p.2.isConstOf kind) do
        throwError "random argument does not match its call-signature dtype"
      if result.any (fun p => p.1==name) then throwError "duplicate random argument"
      result:=result++[(name,← elabTermEnsuringType value (mkConst type))]
    return result
  let words ← read stx[8] ``ScalarType.fp32 ``Float32Bits
  let integers ← read stx[12] ``ScalarType.i64 ``Int
  let mut sizes : List (Name × Expr × Expr) := []
  for entry in stx[16].getSepArgs do
    let `(tritonCallBinding| $name:ident := $proof:term) := entry | throwError "checked arange argument required"
    let key:=name.getId.eraseMacroScopes
    unless constexpr.contains key do throwError "arange argument must be a constexpr parameter"
    if sizes.any (fun p => p.1==key) then throwError "duplicate arange argument"
    let value ← elabTermEnsuringType name (mkConst ``Nat)
    let proof ← elabTermEnsuringType proof (← mkAppM ``Random.ArangeSizeDomain #[value])
    sizes:=sizes++[(key,value,proof)]
  let mut helpers : List Name := []
  for entry in stx[20].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := entry | throwError "named uniform helper required"
    let helper ← elabTermEnsuringType value (mkConst ``UniformJitFunction)
    let .lit (.strVal parameter) ← whnf (← mkAppM ``UniformJitFunction.parameter #[helper])
      | throwError "uniform helper parameter must be static"
    let .lit (.strVal source) ← whnf (← mkAppM ``UniformJitFunction.source #[helper])
      | throwError "uniform helper source must be static"
    UniformJitFunctions.validate parameter source
    helpers:=helpers++[name.getId.eraseMacroScopes]
  let names:=words.map Prod.fst++integers.map Prod.fst++sizes.map (·.1)++helpers
  unless names.eraseDups.length==names.length do throwError "random call bindings must be distinct"
  let occupied:=scalarKinds.map Prod.fst++constexpr
  if helpers.any occupied.contains then throwError "uniform helper shadows a parameter"
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source => pure source | .error message => throwError "{message}"
  let source:=escapeTo <| PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "complete source body required"
  for statement in stmts do
    rejectWrites (words.map Prod.fst++sizes.map (·.1)++helpers) statement.raw
    checkIntegerWrites (integers.map Prod.fst) statement.raw
  let values:=words++integers++sizes.flatMap (fun (_,n,h) => [(Name.anonymous,n),(Name.anonymous,h)])
  let rec freeze (todo : List (Name × Expr)) (bound : List Ident) : TermElabM Expr := do
    match todo with
    | (_,value)::tail =>
      let fresh ← mkFreshId
      let name:=Name.mkSimple ("veritile_random_arg_"++(fresh.toString.map fun c => if c.isAlphanum then c else '_'))
      if (ids parsed++ids stx).contains name then throwError "random binding hygiene collision"
      withLetDecl name (← inferType value) value fun captured => do
        let result ← freeze tail (bound++[mkIdent name])
        mkLetFVars #[captured] result
    | [] =>
      let mut before : Array (TSyntax `tritonStmt) := #[]
      for helper in helpers do
        let helper:=mkIdent helper
        before:=before.push (← `(tritonStmt| tl.source_helper $helper:ident = tl.uint_to_uniform_float))
      for i in [:words.length] do
        let name : Ident := if words[i]!.1==`to then
          ⟨Syntax.ident .none "«to»" `to []⟩ else mkIdent words[i]!.1
        let value:=bound[i]!
        before:=before.push (← `(tritonStmt| $name:ident = tl.bitcast($(($value:ident).bits.toNat), tl.float32)))
      for i in [:integers.length] do
        let name:=mkIdent integers[i]!.1
        let value:=bound[words.length+i]!
        before:=before.push (← `(tritonStmt| $name:ident = $(($value:ident : Int))))
        before:=before.push (← `(tritonStmt| tl.source_integer $name:ident = tl.int64))
      for i in [:sizes.length] do
        let name:=mkIdent sizes[i]!.1
        let n:=bound[words.length+integers.length+2*i]!
        let h:=bound[words.length+integers.length+2*i+1]!
        before:=before.push (← `(tritonStmt| $name:ident = $(_root_.VeriTile.Triton.Random.i32ArangeSize $n:ident $h:ident)))
      let profile ← exprToSyntax profile
      let integers: Array (TSyntax `term):=integers.toArray.map (fun p => ⟨Syntax.mkStrLit p.1.toString⟩)
      let sizes: Array (TSyntax `term):=sizes.toArray.map (fun p => ⟨Syntax.mkStrLit p.1.toString⟩)
      let profile ← `(withoutArguments $profile [$integers,*] [$sizes,*])
      let bindings : Array (TSyntax `tritonCallBinding):=stx[4].getSepArgs.map (⟨·⟩)
      let call ← `(with_triton_call $profile call_args [$bindings,*] in triton {})
      let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" ((before++stmts).toList.map (PythonFullConstructors.render ·.raw))++"\n}")
      let call:=call.raw.setArgs (call.raw.getArgs.set! (call.raw.getArgs.size-1) raw)
      PythonFullConstructors.elaborateInvocation call expected
  freeze values []

end VeriTile.Triton.DSL.PhiloxCallArguments
