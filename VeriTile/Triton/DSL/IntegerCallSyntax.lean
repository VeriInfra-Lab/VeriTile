import VeriTile.Triton.DSL.IntegerCallArguments
import VeriTile.Triton.DSL.StaticRanges
import VeriTile.Triton.DSL.PythonShapeTuples
import VeriTile.Triton.DSL.PythonFullConstructors

/-! Compose checked integer/arange invocation with Python shape constructors
and natural constexpr static-loop bounds. Only constructor spelling and the
static loop bound are lowered; every loop-body statement remains, including a
source reassignment of the static counter. No aligned-arange rewrite is used.
-/
namespace VeriTile.Triton.DSL.IntegerCallSyntax
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr names must form a static list"

private partial def natural (values : List (Name × Ident)) (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) := do
  match e with
  | `(tritonExpr| $n:num) => `(($n:num : Nat))
  | `(tritonExpr| $id:ident) =>
    let some (_,value):=values.find? (·.1==id.getId.eraseMacroScopes)
      | Macro.throwErrorAt id "static_range bound requires declared natural constexprs"
    `(($value:ident : Nat))
  | `(tritonExpr| ($a:tritonExpr)) => natural values a
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
    let a ← natural values a; let b ← natural values b; `($a + $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
    let a ← natural values a; let b ← natural values b; `($a * $b)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
    let a ← natural values a; let b ← natural values b; `($a / $b)
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
    let a ← natural values a; let b ← natural values b; `(($a + $b - 1) / $b)
  | _ => Macro.throwErrorAt e "unsupported natural constexpr static_range bound"

private partial def targetNames (raw : Syntax) : Option (List Name) := do
  if raw.isIdent then return [raw.getId.eraseMacroScopes]
  if raw.isAtom then
    if ["",",",":","tl.constexpr"].contains raw.getAtomVal.trimAscii.toString then return []
    else none
  return (← raw.getArgs.toList.mapM targetNames).flatten

private partial def lower (names : List Name) (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower names values raw[0]!
  if raw.isOfKind ``StaticRanges.staticRange then
    let counter : Ident:=⟨raw[1]⟩
    if names.contains counter.getId.eraseMacroScopes then
      Macro.throwErrorAt counter "static_range counter cannot shadow a constexpr"
    let bound ← natural values ⟨raw[4]⟩
    let body : Array (TSyntax `tritonStmt) ← raw[7].getArgs.mapM fun s=>return ⟨← lower names values s⟩
    -- Keep the ordinary loop spelling through typed load/store lowering.
    -- CallSignature specializes this known natural bound to the static IR last.
    return (← `(tritonStmt| for $counter:ident in range($($bound)) { $body* })).raw
  -- Inspect only syntactic assignment targets, never keyword argument values.
  let args:=raw.getArgs.toList
  for op in ["=",":=","+=","-=","*=","/="] do
    if let some pos:=args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==op) then
      let before:=args.take pos
      if before.head?.any Syntax.isIdent then
        if let some targets:=(before.mapM targetNames).map List.flatten then
          for name in targets do
            if names.contains name then
              Macro.throwErrorAt raw "static_range constexprs must remain immutable"
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      Macro.throwErrorAt counter "a loop cannot shadow a static_range constexpr"
  if let `(tritonExpr| tl.zeros([$dims:tritonExpr,*], dtype=$dtype:tritonDType)) := raw then
    let dims ← dims.getElems.mapM fun d=>return (⟨← lower names values d.raw⟩ : TSyntax `tritonExpr)
    return (← `(tritonExpr| tl.zeros([$dims,*], $dtype:tritonDType))).raw
  return raw.setArgs (← raw.getArgs.mapM (lower names values))

private def prepare (stx : Syntax) (names : List Name) (values : List (Name × Ident)) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw "{message}"
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source))
  let parsed ← match Parser.runParserCategory (Parser.parserExtension.activateScoped (← getEnv)
      `VeriTile.Triton.DSL.StaticRanges) `term ("triton {\n"++source++"\n}") with
    | .ok source=>pure source | .error message=>throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := (← liftMacroM (lower names values parsed))
    | throwError "expected a complete function body"
  let source:="{\n"++String.intercalate "\n" (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))

@[term_elab IntegerCallArguments.withIntegers]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut arguments : List (Name × Expr):=[]
  for item in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf item | throwError "constexpr names must be literal"
    arguments:=arguments++[(Name.mkSimple name,← elabTerm (mkIdent (Name.mkSimple name)) none)]
  -- Freeze caller constexpr values before the arange adapter substitutes source
  -- names. Its source-expression substitution must never enter a Lean bound.
  let rec freeze (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | [] => do
      let prepared ← prepare stx (arguments.map Prod.fst) values
      IntegerCallArguments.elaborateInvocation prepared expected
    | (name,value)::tail =>
      let mut fresh:=Name.anonymous
      repeat
        let id ← mkFreshId
        fresh:=Name.mkSimple ("veritile_static_call_"++id.toString.map (fun c=>if c.isAlphanum then c else '_'))
        unless (stx.getArgs.back!.getAtomVal.splitOn fresh.toString).length>1 do break
      withLetDecl fresh (← inferType value) value fun bound=>do
        let result ← freeze tail (values++[(name,mkIdent fresh)])
        mkLetFVars #[bound] result
  freeze arguments []
end VeriTile.Triton.DSL.IntegerCallSyntax
