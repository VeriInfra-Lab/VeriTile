import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.PythonInlineSuites

/-! Explicit source for unary, pure, single-return JIT helpers. Both the helper
and caller bodies remain pasted Python. Calls evaluate the argument and return
in fresh registers before the enclosing arithmetic, preserving the callee's
precision instead of propagating the caller's result dtype into its body.
Memory operations, free variables, nested helpers and control-flow call sites
are deliberately rejected by this initial supported subset. -/
namespace VeriTile.Triton

structure PureJitFunction where
  parameter : String
  source : String

namespace DSL.PureJitFunctions
open Lean Elab Term Meta

syntax (name := pureFunction) "triton_function " "(" ident ")" RawSourceText.body : term
syntax (name := withFunctions) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " pure_functions " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term

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
private def call? (raw : Syntax) : Option (Name × Syntax) := do
  let a := raw.getArgs
  if a.size==4 && a[1]!.isAtom && a[1]!.getAtomVal=="(" then
    let name := if a[0]!.isIdent then a[0]!.getId.eraseMacroScopes
      else a[0]!.getAtomVal.toName
    return (name,a[2]!)
  if a.size==3 && a[0]!.isAtom then
    let text := a[0]!.getAtomVal
    if text.endsWith "(" then return ((text.dropEnd 1).toString.toName,a[1]!)
  none

private def parseReturn (parameter : Name) (source : String) : TermElabM Syntax := do
  let source := source.trimAscii.toString
  unless source.startsWith "return " do throwError "pure JIT helper requires one return expression"
  let expression := (source.drop 7).toString
  let parsed ← match Parser.runParserCategory (← getEnv) `tritonExpr expression with
    | .ok value => pure value
    | .error _ => throwError "pure JIT helper requires one return expression"
  unless parameter.isStr && parameter.getPrefix.isAnonymous &&
      !["tl","triton","libdevice"].contains parameter.toString do
    throwError "pure JIT helper requires an unreserved simple parameter"
  let allowed := ["tl.sigmoid","tl.exp","tl.sqrt","tl.abs","tl.where","tl.maximum","tl.minimum"]
  let rec checkCalls (raw : Syntax) (fuel : Nat) : TermElabM Unit := do
    match fuel with
    | 0 => throwError "pure JIT helper expression is too deeply nested"
    | fuel+1 =>
      if let some (name,_) := call? raw then
        unless allowed.contains name.toString do
          throwError "pure JIT helper cannot call other functions"
      for child in raw.getArgs do checkCalls child fuel
  checkCalls parsed 256
  for id in ids parsed do
    unless id==parameter || allowed.contains id.toString do
      throwError "pure JIT helper cannot reference free names or other functions"
  for token in tokens parsed do
    if token.startsWith "tl." && !(allowed.any fun name =>
        token==name || token==name++"(" ) then
      throwError "unsupported operation in pure JIT helper"
    if token.contains '$' then throwError "Lean expressions cannot occur in a pasted JIT helper"
  return parsed

@[term_elab pureFunction] def elaborateFunction : TermElab := fun stx expected => do
  let parameter := stx[2].getId.eraseMacroScopes
  let source ← sourceText stx.getArgs.back!
  let _ ← parseReturn parameter source
  let value := mkApp2 (mkConst ``PureJitFunction.mk) (mkStrLit parameter.toString) (mkStrLit source)
  return ← ensureHasType expected value

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal value) ← whnf value | throwError "pure JIT source must be statically available"
  return value

private partial def substitute (parameter : Name) (argument : Ident) (raw : Syntax) : Syntax :=
  if raw.isIdent && raw.getId.eraseMacroScopes==parameter then argument.raw
  else raw.setArgs (raw.getArgs.map (substitute parameter argument))

private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_jit_"++toString n)) do n:=n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_jit_"++toString n))
private partial def lower (functions : List (Name × Name × Syntax)) (used : List Name) (raw : Syntax) : StateRefT Nat TermElabM (Array (TSyntax `tritonStmt) × Syntax) := do
  if let some (name,arg) := call? raw then
    if let some (_,parameter,body) := functions.find? (fun f => f.1==name) then
      let arg : TSyntax `tritonExpr := ⟨arg⟩
      unless (ids arg).length==1 && (tokens arg).length==1 do
        throwError "pure JIT call arguments must be register names"
      let argument ← fresh used
      let result ← fresh used
      let body : TSyntax `tritonExpr := ⟨substitute parameter argument body⟩
      let prepended := #[← `(tritonStmt| $argument:ident = $arg:tritonExpr),
        ← `(tritonStmt| $result:ident = $body:tritonExpr)]
      let value ← `(tritonExpr| $result:ident)
      return (prepended,value.raw)
  if raw.isIdent && functions.any (fun f => f.1==raw.getId.eraseMacroScopes) then
    throwError "pure JIT functions cannot be rebound, aliased or escape direct calls"
  let mut prepended := #[]
  let mut args := #[]
  for child in raw.getArgs do
    let (before,child) ← lower functions used child
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
  let text := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++text++"\n}") with
    | .ok body => pure body
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let arguments ← argumentNames profile
  let mut functions : List (Name × Name × Syntax) := []
  for binding in stx[8].getSepArgs do
    let `(tritonCallBinding| $id:ident := $value:term) := binding
      | throwError "expected a named pure JIT function"
    let name := id.getId.eraseMacroScopes
    if functions.any (fun f => f.1==name) then throwError "duplicate pure JIT function binding"
    if arguments.contains name.toString then throwError "pure JIT function captures a caller parameter"
    if !name.getPrefix.isAnonymous || ["tl","triton","libdevice"].contains name.toString then
      throwError "a helper cannot replace a language namespace"
    let value ← elabTermEnsuringType value (mkConst ``PureJitFunction)
    let parameter ← stringValue (← mkAppM ``PureJitFunction.parameter #[value])
    let source ← stringValue (← mkAppM ``PureJitFunction.source #[value])
    let body ← parseReturn parameter.toName source
    functions := functions ++ [(name,parameter.toName,body)]
  let used := ids parsed ++ ids stx ++ arguments.map String.toName
  let mut next := 0
  let mut lowered := #[]
  for stmt in stmts do
    let ((prepended,body),n) ← (lower functions used stmt.raw).run next
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

end DSL.PureJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.PureJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceProcedures.sourceParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``PureJitFunction do throwUnsupportedSyntax
  let parameters := stx[2].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let constants := stx[6].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  unless stx[8].getArgs.isEmpty do
    throwError "symbolic FP32 constants require a SourceProcedure"
  let #[x] := parameters | throwError "unary helper requires one parameter"
  unless constants.isEmpty do throwError "unary helper requires an empty constexpr list"
  let body ← `(triton_function ($x:ident) {})
  elaborateFunction (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.PureJitFunctions
