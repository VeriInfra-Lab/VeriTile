import VeriTile.Triton.DSL.SourceSyntax
import VeriTile.Triton.DSL.SignedRankOneBlockPointers
import VeriTile.Triton.DSL.BinaryPureJitFunctions

/-! Copied binary libdevice-power helpers and their source import binding.
Both arguments are captured at the call site, before the helper's explicit
FP32 cast and power operation. Rank-one block pointers retain the existing
signed-base/stride lowering. No caller expression enters the callee's scope. -/
namespace VeriTile.Triton
structure ImportedBinaryJitFunction where
  first : String
  second : String
  source : String
  imports : List (String × String)

namespace DSL.ImportedBinaryJitFunctions
open Lean Elab Term Meta
syntax (name := importedFunction) "with_triton_imports " "[" tritonCallBinding,* "]"
  " in " "triton_function " "(" ident ", " ident ")" RawSourceText.body : term
syntax (name := withFunctions) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " signed_pointer_offsets " "[" tritonCallBinding,* "]"
  " binary_functions " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private def textValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "helper metadata must be static strings"
  return text
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "helper metadata must be a static list"
private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifiers
private def sourceText (raw : Syntax) : TermElabM String :=
  match RawSourceText.normalize raw.getAtomVal with
  | .ok text => pure text | .error message => throwErrorAt raw "{message}"
private def simple (name : Name) : Bool :=
  name.isStr && name.getPrefix.isAnonymous &&
    (match name.toString.toList with
      | [] => Bool.false
      | first::rest => (first.isAlpha || first == '_') &&
          rest.all (fun c => c.isAlphanum || c == '_')) &&
    !["tl","triton","libdevice"].contains name.toString

/-- This closed supported helper has exactly its two arguments and one
explicit source cast. The external alias is supplied outside its body. -/
private def checkedReturn (first second : Name) (source : String)
    (imports : List (String × String)) : TermElabM Syntax := do
  unless simple first && simple second && first != second do
    throwError "binary helper requires two distinct unreserved parameters"
  let [(aliasName,target)] := imports | throwError "power helper requires one explicit imported alias"
  unless simple aliasName.toName && aliasName.toName != first && aliasName.toName != second &&
      target == "triton.language.extra.cuda.libdevice.pow" do
    throwError "unsupported binary helper extern binding"
  let compact := String.ofList (source.toList.filter (fun c => !c.isWhitespace))
  unless compact == "return"++aliasName++"("++first.toString++".to(tl.float32),"++second.toString++")" do
    throwError "power helper requires its complete copied cast-and-power return expression"
  let a := mkIdent first
  let b := mkIdent second
  return (← `(tritonExpr| tl.extra.cuda.libdevice.pow(($a:ident).to(tl.float32), $b:ident))).raw

@[term_elab importedFunction] def elaborateFunction : TermElab := fun stx expected => do
  let first := stx[7].getId.eraseMacroScopes
  let second := stx[9].getId.eraseMacroScopes
  let source ← sourceText stx.getArgs.back!
  let mut imports := []
  for binding in stx[2].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding | throwError "expected an extern binding"
    let value ← elabTermEnsuringType value (mkConst ``String)
    imports := imports ++ [(name.getId.eraseMacroScopes.toString,← textValue value)]
  let _ ← checkedReturn first second source imports
  let pairs ← imports.mapM fun (a,b) => mkAppM ``Prod.mk #[mkStrLit a,mkStrLit b]
  let importValue ← mkListLit (mkApp2 (mkConst ``Prod [levelZero,levelZero])
    (mkConst ``String) (mkConst ``String)) pairs
  ensureHasType expected (mkApp4 (mkConst ``ImportedBinaryJitFunction.mk)
    (mkStrLit first.toString) (mkStrLit second.toString) (mkStrLit source) importValue)

private abbrev Functions := List (Name × Name × Name × Syntax)
private def rejectFunctions (functions : Functions) (raw : Syntax) : TermElabM Unit := do
  if (identifiers raw).any (fun name => functions.any (·.1 == name)) then
    throwErrorAt raw "binary JIT helpers require a complete plain assignment with scalar/tensor register arguments"
private def fresh (used : List Name) : StateRefT Nat TermElabM Ident := do
  let mut n ← get
  while used.contains (Name.mkSimple ("__veritile_binary_power_"++toString n)) do n := n+1
  set (n+1)
  return mkIdent (Name.mkSimple ("__veritile_binary_power_"++toString n))
private partial def lower (functions : Functions) (used : List Name)
    (statements : Array (TSyntax `tritonStmt)) : StateRefT Nat TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut result := #[]
  for statement in statements do
    if let `(tritonStmt| $target:ident = $callee:ident($a:tritonExpr, $b:tritonExpr)) := statement then
      if let some (_,_,_,_) := functions.find? (·.1 == callee.getId.eraseMacroScopes) then
        let `(tritonExpr| $a:ident) := a | throwError "binary helper arguments must be register names"
        let `(tritonExpr| $b:ident) := b | throwError "binary helper arguments must be register names"
        rejectFunctions functions target.raw
        rejectFunctions functions a.raw
        rejectFunctions functions b.raw
        let input ← fresh used
        let exponent ← fresh used
        let output ← fresh used
        -- The validated helper casts only its first argument before power.
        -- Place that sole cast on its capture; both arguments are register
        -- reads, so this retains every FP operation and its operands.
        let value ← `(tritonExpr| tl.extra.cuda.libdevice.pow($input:ident, $exponent:ident))
        result := result ++ #[← `(tritonStmt| $input:ident = ($a:ident).to(tl.float32)),
          ← `(tritonStmt| $exponent:ident = $b:ident),
          ← `(tritonStmt| $output:ident = $value:tritonExpr),
          ← `(tritonStmt| $target:ident = $output:ident)]
        continue
    match statement with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      rejectFunctions functions condition.raw
      let yes ← lower functions used yes
      let no ← lower functions used no
      result := result.push (← `(tritonStmt| if $condition { $yes* } else { $no* }))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      rejectFunctions functions condition.raw
      let yes ← lower functions used yes
      result := result.push (← `(tritonStmt| if $condition { $yes* }))
    | `(tritonStmt| for $counter:ident in range(0, $stop:tritonExpr) { $body:tritonStmt* }) =>
      rejectFunctions functions counter.raw
      rejectFunctions functions stop.raw
      let body ← lower functions used body
      result := result.push (← `(tritonStmt| for $counter:ident in range(0, $stop) { $body* }))
    | `(tritonStmt| for $counter:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
      rejectFunctions functions counter.raw
      rejectFunctions functions start.raw
      rejectFunctions functions stop.raw
      let body ← lower functions used body
      result := result.push (← `(tritonStmt| for $counter:ident in range($start, $stop) { $body* }))
    | _ =>
      rejectFunctions functions statement.raw
      result := result.push statement
  return result

@[term_elab withFunctions] def elaborateInvocation : TermElab := fun stx expected => do
  let prepared ← PythonFullConstructors.prepareSource stx
  let source ← sourceText prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value => pure value | .error message => throwError "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete copied caller"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut declared := []
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,``CallSignature.scalars] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      declared := declared ++ [(← textValue (← mkAppM ``Prod.fst #[pair])).toName]
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    declared := declared ++ [(← textValue name).toName]
  let mut functions : Functions := []
  for binding in stx[12].getSepArgs do
    let `(tritonCallBinding| $name:ident := $value:term) := binding | throwError "expected a helper binding"
    let key := name.getId.eraseMacroScopes
    unless simple key && !declared.contains key && !functions.any (·.1 == key) do
      throwError "duplicate or captured binary helper name"
    let value ← elabTermEnsuringType value (mkConst ``ImportedBinaryJitFunction)
    let first ← textValue (← mkAppM ``ImportedBinaryJitFunction.first #[value])
    let second ← textValue (← mkAppM ``ImportedBinaryJitFunction.second #[value])
    let source ← textValue (← mkAppM ``ImportedBinaryJitFunction.source #[value])
    let mut imports := []
    for pair in ← listValues (← mkAppM ``ImportedBinaryJitFunction.imports #[value]) do
      imports := imports ++ [((← textValue (← mkAppM ``Prod.fst #[pair])),(← textValue (← mkAppM ``Prod.snd #[pair])))]
    let body ← checkedReturn first.toName second.toName source imports
    functions := functions ++ [(key,first.toName,second.toName,body)]
  let used := identifiers stx ++ identifiers parsed ++ declared
  let (statements,_) ← (lower functions used statements).run 0
  let raw := Syntax.atom .none ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let base := mkNode ``SignedRankOneBlockPointers.withSignedBlockCall
    #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[6],stx[7],stx[8],stx[9],
      Syntax.atom .none "pure_functions",stx[11],mkNullNode #[],stx[13],stx[14],stx[15],raw]
  SignedRankOneBlockPointers.elaborateInvocation base expected
end DSL.ImportedBinaryJitFunctions
end VeriTile.Triton

namespace VeriTile.Triton.DSL.ImportedBinaryJitFunctions
open Lean Elab Term Meta

/-- Forward the common source block to this helper's established elaborator. -/
@[term_elab SourceSyntax.importedParams]
def elaborateSourceParams : TermElab := fun stx expected => do
  let some type := expected | throwUnsupportedSyntax
  unless (← whnf type).isConstOf ``ImportedBinaryJitFunction do throwUnsupportedSyntax
  let imports := stx[2].getSepArgs.map (⟨·⟩ : Syntax → TSyntax `tritonCallBinding)
  let parameters := stx[7].getSepArgs.map (⟨·⟩ : Syntax → Ident)
  let #[x, y] := parameters | throwError "imported binary helper requires two parameters"
  let body ← `(with_triton_imports [$imports,*] in triton_function ($x:ident, $y:ident) {})
  elaborateFunction (SourceSyntax.attachBody body.raw stx) expected

end VeriTile.Triton.DSL.ImportedBinaryJitFunctions
