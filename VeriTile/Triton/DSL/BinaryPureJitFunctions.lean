import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.PythonShapeTuples

/-! Two-parameter, single-return pure JIT helpers. The original helper and
caller bodies remain pasted Python. Each argument is captured in a fresh
register before evaluating the helper result; repeated calls get independent
captures. This arithmetic subset permits cdiv and ordinary source operators,
and rejects memory, free names, rebinding and control-flow call sites. Calls
require declared i64 scalars with Int bindings (or integer constexpr values)
and a signed argument in the returned expression. The internal int64 type
assertion is an identity on that already-signed expression; it emits no cast
operation, as checked by independent typed-program equality tests. -/
namespace VeriTile.Triton

structure BinaryPureJitFunction where
  first : String
  second : String
  source : String

namespace DSL.BinaryPureJitFunctions
open Lean Elab Term Meta

syntax (name := pureFunction) "triton_function " "(" ident ", " ident ")" RawSourceText.body : term
syntax (name := withFunctions) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " binary_functions " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term

syntax (name := binaryHelperCall) ident "(" tritonExpr ", " tritonExpr ")" : tritonExpr

private def sourceText (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok source => pure source
  | .error message => throwErrorAt raw "{message}"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap ids

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args =>
      if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

/-- Builtin parser spellings such as silu(x) and generic helper(x) are both
recognized as calls; their implementation is supplied only by the binding. -/
private def call? (raw : Syntax) : Option (Name × Syntax × Syntax) := do
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  let `(tritonExpr| $callee:ident($first:tritonExpr, $second:tritonExpr)) := expression | none
  return (callee.getId.eraseMacroScopes,first.raw,second.raw)

private def parseReturn (parameter partner : Name) (source : String) : TermElabM Syntax := do
  let source := source.trimAscii.toString
  unless source.startsWith "return " do throwError "pure JIT helper requires one return expression"
  let expression := (source.drop 7).toString
  let parsed ← match Parser.runParserCategory (← getEnv) `tritonExpr expression with
    | .ok value => pure value
    | .error _ => throwError "pure JIT helper requires one return expression"
  unless parameter.isStr && parameter.getPrefix.isAnonymous &&
      !["tl","triton","libdevice"].contains parameter.toString do
    throwError "pure JIT helper requires an unreserved simple parameter"
  unless partner.isStr && partner.getPrefix.isAnonymous && partner != parameter &&
      !["tl","triton","libdevice"].contains partner.toString do
    throwError "binary JIT helper requires two distinct unreserved parameters"
  let rec integerExpression (raw : TSyntax `tritonExpr) (fuel : Nat) : Bool :=
    match fuel with
    | 0 => Bool.false
    | fuel+1 => match raw with
      | `(tritonExpr| $id:ident) => id.getId.eraseMacroScopes==parameter || id.getId.eraseMacroScopes==partner
      | `(tritonExpr| $_:num) => Bool.true
      | `(tritonExpr| ($value:tritonExpr)) => integerExpression value fuel
      | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
      | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
      | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
      | `(tritonExpr| $a:tritonExpr // $b:tritonExpr)
      | `(tritonExpr| $a:tritonExpr % $b:tritonExpr)
      | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
        integerExpression a fuel && integerExpression b fuel
      | _ => Bool.false
  unless integerExpression ⟨parsed⟩ 256 do
    throwError "binary JIT helper supports only its two parameters, integer literals and integer arithmetic"
  return parsed

@[term_elab pureFunction] def elaborateFunction : TermElab := fun stx expected => do
  let parameter := stx[2].getId.eraseMacroScopes
  let partner := stx[4].getId.eraseMacroScopes
  let source ← sourceText stx.getArgs.back!
  let _ ← parseReturn parameter partner source
  let value := mkApp3 (mkConst ``BinaryPureJitFunction.mk) (mkStrLit parameter.toString) (mkStrLit partner.toString) (mkStrLit source)
  return ← ensureHasType expected value

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal value) ← whnf value | throwError "pure JIT source must be statically available"
  return value

private partial def substitute (parameter : Name) (argument : Ident) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes==parameter then argument.raw
  else raw.setArgs (raw.getArgs.map (substitute parameter argument))

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for op in ["=",":=","+=","-=","*=","/="] do
    if args.head?.any (·.isIdent) then
      if let some i := args.findIdx? (fun s=>s.isAtom && s.getAtomVal.trimAscii.toString==op) then
        if ((args.take i).flatMap ids).any names.contains then
          throwError "binary JIT argument typing requires immutable integer call and constexpr parameters"
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        (ids b).any names.contains then throwError "a loop cannot shadow a binary JIT integer parameter"
  for arg in raw.getArgs do rejectWrites names arg

private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_binary_jit_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_binary_jit_"++toString n))
private partial def lower (functions : List (Name × Name × Name × Syntax)) (used integral signed : List Name) (raw : Syntax) : StateRefT Nat TermElabM (Array (TSyntax `tritonStmt) × Syntax) := do
  if let some (name,arg,arg2) := call? raw then
    if let some (_,parameter,partner,body) := functions.find? (fun f => f.1==name) then
      let arg : TSyntax `tritonExpr := ⟨arg⟩
      unless (ids arg).length==1 && (tokens arg).length==1 do
        throwError "pure JIT call arguments must be register names"
      let arg2 : TSyntax `tritonExpr := ⟨arg2⟩
      unless (ids arg2).length==1 && (tokens arg2).length==1 do
        throwError "binary JIT call arguments must be register or constexpr names"
      let firstName := (ids arg).head!
      let secondName := (ids arg2).head!
      unless integral.contains firstName && integral.contains secondName do
        throwError "binary JIT arguments require declared integer scalars or integer constexpr values"
      unless (signed.contains firstName && (ids body).contains parameter) ||
          (signed.contains secondName && (ids body).contains partner) do
        throwError "binary JIT result requires a signed caller argument in its integer expression"
      let argument ← fresh used
      let argument2 ← fresh used
      let result ← fresh used
      let body : TSyntax `tritonExpr := ⟨substitute partner argument2 (substitute parameter argument body)⟩
      let prepended := #[← `(tritonStmt| $argument:ident = $arg:tritonExpr),
        ← `(tritonStmt| $argument2:ident = $arg2:tritonExpr),
        ← `(tritonStmt| $result:ident = tl.cast($body:tritonExpr, tl.int64))]
      let value ← `(tritonExpr| $result:ident)
      return (prepended,value.raw)
  if raw.isIdent && functions.any (fun f => f.1==raw.getId.eraseMacroScopes) then
    throwError "pure JIT functions cannot be rebound, aliased or escape direct calls"
  let mut prepended := #[]
  let mut args := #[]
  for child in raw.getArgs do
    let (before,child) ← lower functions used integral signed child
    prepended:=prepended++before
    args:=args.push child
  return (prepended,raw.setArgs args)

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "call profile must have statically known argument names"

private def argumentNames (value : Expr) : TermElabM (List String) := do
  let mut names := []
  for field in [``CallSignature.pointers, ``CallSignature.floatingPointers, ``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[value]) do
      names:=names++[← stringValue (← mkAppM ``Prod.fst #[pair])]
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[value]) do
    names:=names++[← stringValue name]
  return names

@[term_elab withFunctions] def elaborateInvocation : TermElab := fun stx expected => do
  let source ← sourceText stx.getArgs.back!
  let source := PythonShapeTuples.normalize source
  let text := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body => pure body
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let arguments ← argumentNames profile
  let mut functions : List (Name × Name × Name × Syntax) := []
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding
      | throwError "expected a named pure JIT function"
    let name := id.getId.eraseMacroScopes
    if functions.any (fun f => f.1==name) then throwError "duplicate pure JIT function binding"
    if arguments.contains name.toString then throwError "pure JIT function captures a caller parameter"
    if !name.getPrefix.isAnonymous || ["tl","triton","libdevice"].contains name.toString then
      throwError "a helper cannot replace a language namespace"
    let value ← elabTermEnsuringType value (mkConst ``BinaryPureJitFunction)
    let parameter ← stringValue (← mkAppM ``BinaryPureJitFunction.first #[value])
    let partner ← stringValue (← mkAppM ``BinaryPureJitFunction.second #[value])
    let source ← stringValue (← mkAppM ``BinaryPureJitFunction.source #[value])
    let body ← parseReturn parameter.toName partner.toName source
    functions := functions ++ [(name,parameter.toName,partner.toName,body)]
  let mut integral : List Name := []
  let mut signed : List Name := []
  for pair in ← listValues (← mkAppM ``CallSignature.scalars #[profile]) do
    let name := (← stringValue (← mkAppM ``Prod.fst #[pair])).toName
    let kind ← whnf (← mkAppM ``Prod.snd #[pair])
    if kind.isConstOf ``ScalarType.i64 then
      integral:=integral++[name]
  for binding in stx[4].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding | throwError "expected scalar binding"
    let name := id.getId.eraseMacroScopes
    if integral.contains name then
      let value ← elabTerm value none
      let type ← whnf (← inferType value)
      if type.isConstOf ``Int then signed:=signed++[name]
  for raw in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let name := (← stringValue raw).toName
    let value ← elabTerm (mkIdent name) none
    let type ← whnf (← inferType value)
    if type.isConstOf ``Nat || type.isConstOf ``Int then integral:=integral++[name]
    if type.isConstOf ``Int then signed:=signed++[name]
  rejectWrites integral parsed
  let used := ids parsed ++ ids stx ++ arguments.map String.toName
  let mut next := 0
  let mut lowered := #[]
  for stmt in stmts do
    let ((prepended,body),n) ← (lower functions used integral signed stmt.raw).run next
    next:=n
    unless prepended.isEmpty do
      let `(tritonStmt| $_:ident = $_:tritonExpr) := stmt
        | throwError "pure JIT calls currently require a plain assignment"
      if (tokens stmt).any (fun t => ["if","else","tl.load(","tl.store("].contains t) then
        throwError "pure JIT calls cannot share a statement with conditional expressions or memory operations"
    lowered:=(lowered++prepended).push (⟨body⟩ : TSyntax `tritonStmt)
  let body := Syntax.atom .none ("{\n"++String.intercalate "\n" (lowered.toList.map (render ·.raw))++"\n}")
  let base := mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],body]
  PythonInlineSuites.elaborateInvocation base expected

end DSL.BinaryPureJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.BinaryPureJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceProcedures.sourceParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``BinaryPureJitFunction do throwUnsupportedSyntax
  let parameters := stx[2].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let constants := stx[6].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  unless stx[8].getArgs.isEmpty do
    throwError "symbolic FP32 constants require a SourceProcedure"
  let #[x, y] := parameters | throwError "binary helper requires two parameters"
  unless constants.isEmpty do throwError "binary helper requires an empty constexpr list"
  let body ← `(triton_function ($x:ident, $y:ident) {})
  elaborateFunction (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.BinaryPureJitFunctions
