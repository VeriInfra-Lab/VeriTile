import VeriTile.Triton.DSL.PythonInlineSuites
import VeriTile.Triton.DSL.RandomProfile
import Lean.Meta.Tactic.Cases

/-! Explicit integer ABI values and checked symbolic arange sizes supplied by
a call. The copied body contains neither source-integer annotations nor
representability adapters. Integer binding remains an invocation prefix; an
arange adapter is the identity justified by the caller's existing shape proof.
No RNG primitive, seed bit, offset arithmetic or floating operation is replaced. -/
namespace VeriTile.Triton

inductive IntegerArgument where
  | i32 (value : Int) | i64 (value : Int)
  | u32 (value : Nat) | u64 (value : Nat)

namespace DSL.IntegerCallArguments
open Lean Elab Term Meta

def withoutScalars (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with scalars := profile.scalars.filter (fun arg => !names.contains arg.1)}

syntax (name := withIntegers) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " integer_args " "[" tritonCallBinding,* "]"
  " arange_args " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "call argument names must be statically known"

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "call argument name must be literal"
  return text

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      for lhs in args.take i do
        if (ids lhs).any names.contains then
          throwError "integer and arange call arguments must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then
      throwError "a loop cannot shadow an integer or arange call argument"
  for arg in raw.getArgs do rejectWrites names arg

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

private partial def substitute (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) : MacroM Syntax := do
  if raw.isIdent then
    if let some (_,value) := values.find? (fun x => x.1==raw.getId.eraseMacroScopes) then return value.raw
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,value) := values.find? (fun x => x.1==id.getId.eraseMacroScopes) then return value.raw
  if let some (receiver,dtype) := methodCast? ⟨raw⟩ then
    let next : TSyntax `tritonExpr := ⟨← substitute values receiver.raw⟩
    return (← `(tritonExpr| ($next).to($dtype:tritonDType))).raw
  return raw.setArgs (← raw.getArgs.mapM (substitute values))

private def integerKind? (value : Expr) : Option (Name × Expr) := do
  let name := value.getAppFn.constName?
  let dtype := if name==some ``IntegerArgument.i32 then `tl.int32
    else if name==some ``IntegerArgument.i64 then `tl.int64
    else if name==some ``IntegerArgument.u32 then `tl.uint32
    else if name==some ``IntegerArgument.u64 then `tl.uint64 else Name.anonymous
  unless !dtype.isAnonymous do none
  return (dtype,value.appArg!)

private partial def compile (profile : Expr) (scalars integers : List (Name × Expr))
    (sizes : List (Name × Expr × Expr)) (parsed : Syntax) (used : List Name)
    (expected : Option Expr) : TermElabM Expr := do
  for (boundName,argument) in integers do
    let argument ← whnf argument
    if let some (dtype,_) := integerKind? argument then
      for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
        let name ← stringValue (← mkAppM ``Prod.fst #[pair])
        if boundName.toString==name then
          let kind ← whnf (← mkAppM ``Prod.snd #[pair])
          unless kind.isConstOf ``ScalarType.integer ||
              (kind.isConstOf ``ScalarType.i32 && dtype==`tl.int32) ||
              (kind.isConstOf ``ScalarType.i64 && dtype==`tl.int64) do
            throwError "integer argument ABI differs from the scalar call type"
      continue
    let .fvar id := argument | throwError "integer_args requires explicit IntegerArgument ABI values"
    let goal ← mkFreshExprSyntheticOpaqueMVar (mkConst ``ComputeKernel)
    let branches ← goal.mvarId!.cases id
    for branch in branches do
      branch.mvarId.withContext do
        let apply := branch.subst.apply
        let result ← compile (apply profile)
          (scalars.map fun (n,v) => (n,apply v))
          (integers.map fun (n,v) => (n,apply v))
          (sizes.map fun (n,v,h) => (n,apply v,apply h)) parsed used
          (some (← branch.mvarId.getType))
        branch.mvarId.assign result
    return ← instantiateMVars goal
  let mut all : List (Name × Expr) := scalars
  for (name,argument) in integers do
    let some (_,value) := integerKind? (← whnf argument) | throwError "unknown integer ABI"
    all:=all++[(name,value)]
  for (_,value,proof) in sizes do all:=all++[(Name.anonymous,value),(Name.anonymous,proof)]
  let rec freeze (todo : List (Name × Expr)) (frozen : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let dtype ← inferType value
      let mut fresh := Name.anonymous
      repeat
        let next ← mkFreshId
        fresh:=Name.mkSimple ("veritile_integer_call_"++(next.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains fresh do break
      withLetDecl fresh dtype value fun bound => do
        let result ← freeze tail (frozen++[(name,mkIdent fresh)])
        mkLetFVars #[bound] result
    | [] =>
      let mut bindings : Array (TSyntax `tritonCallBinding) := #[]
      let mut before : Array (TSyntax `tritonStmt) := #[]
      for (name,_) in scalars do
        let some (_,value) := frozen.find? (fun x => x.1==name) | throwError "missing frozen scalar"
        let id := mkIdent name
        bindings:=bindings.push (← `(tritonCallBinding| $id:ident := $value:ident))
      for (name,argument) in integers do
        let some (dtypeName,value) := integerKind? (← whnf argument) | throwError "unknown integer ABI"
        let some dtype ← liftMacroM <| dtypeSyntaxOfName? dtypeName
          | throwError "unsupported integer ABI"
        let some (_,valueName) := frozen.find? (fun x => x.1==name) | throwError "missing frozen integer"
        let id := mkIdent name
        let signed ← isDefEq (← inferType value) (mkConst ``Int)
        let rhs ← if signed then `(tritonExpr| $(($valueName:ident : Int)))
          else `(tritonExpr| $(($valueName:ident : Nat)))
        before:=before.push (← `(tritonStmt| $id:ident = $rhs:tritonExpr))
        before:=before.push (← `(tritonStmt| tl.source_integer $id:ident = $dtype:tritonDType))
      let mut replacements : List (Name × TSyntax `tritonExpr) := []
      let offset := scalars.length+integers.length
      for i in [:sizes.length] do
        let (name,_,_) := sizes[i]!
        let n := frozen[offset+2*i]!.2
        let h := frozen[offset+2*i+1]!.2
        let value ← `(tritonExpr| $(_root_.VeriTile.Triton.Random.i32ArangeSize $n:ident $h:ident))
        replacements:=replacements++[(name,value)]
      let parsed ← liftMacroM <| substitute replacements parsed
      let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
      let names : Array (TSyntax `term) := integers.toArray.map (fun x => ⟨Syntax.mkStrLit x.1.toString⟩)
      let profile ← exprToSyntax profile
      let profile ← `(term| withoutScalars $profile [$names,*])
      let call ← `(term| with_triton_call $profile call_args [$bindings,*] in triton {})
      let raw := Syntax.atom .none ("{\n"++String.intercalate "\n" ((before++stmts).toList.map (render ·.raw))++"\n}")
      let call := call.raw.setArgs (call.raw.getArgs.set! (call.raw.getArgs.size-1) raw)
      PythonInlineSuites.elaborateInvocation call expected
  freeze all []

@[term_elab withIntegers] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut scalarNames := []
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    scalarNames:=scalarNames++[← stringValue (← mkAppM ``Prod.fst #[pair])]
  let mut constants := []
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    constants:=constants++[← stringValue name]
  let readBindings (raw : Syntax) (integer : Bool) : TermElabM (List (Name × Expr)) := do
    let mut result := []
    for binding in raw.getSepArgs do
      let `(tritonCallBinding| $id:ident := $value:term) := binding | throwError "expected a named argument"
      let name := id.getId.eraseMacroScopes
      unless scalarNames.contains name.toString do throwError "integer call binding is not a scalar parameter"
      if result.any (fun p => p.1==name) then throwError "duplicate call argument binding"
      let value ← if integer then elabTermEnsuringType value (mkConst ``IntegerArgument) else elabTerm value none
      result:=result++[(name,value)]
    return result
  let scalars ← readBindings stx[4] Bool.false
  let integers ← readBindings stx[8] Bool.true
  if (scalars.map Prod.fst).any (integers.map Prod.fst).contains then throwError "duplicate call argument binding"
  let mut sizes := []
  for binding in stx[12].getSepArgs do
    let `(tritonCallBinding| $id:ident := $proof:term) := binding | throwError "expected a checked arange size"
    let name := id.getId.eraseMacroScopes
    unless constants.contains name.toString do throwError "checked arange size must be a constexpr parameter"
    if sizes.any (fun x => x.1==name) then throwError "duplicate arange argument binding"
    let value ← elabTermEnsuringType id (mkConst ``Nat)
    let proof ← elabTermEnsuringType proof (← mkAppM ``Random.ArangeSizeDomain #[value])
    sizes:=sizes++[(name,value,proof)]
  let raw := stx.getArgs.back!
  let text ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwError "{message}"
  let text := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames text))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok value => pure value
    | .error message => throwError "{message}"
  rejectWrites (integers.map Prod.fst++sizes.map fun x => x.1) parsed
  compile profile scalars integers sizes parsed (ids parsed++ids stx) expected

end DSL.IntegerCallArguments
end VeriTile.Triton
