import VeriTile.Triton.DSL.BinaryPowerConstants
import VeriTile.Triton.DSL.IntegerWhereLiterals
import VeriTile.Triton.DSL.PythonFullConstructors
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! Immutable floating constexpr expression trees. Interpretation is separate
from runtime tensor arithmetic: each primitive belongs to the caller's chosen
compile-time numerical model. The mathematical model below is exact Real;
using it does not assert binary64 or IEEE power accuracy. -/
namespace VeriTile.Triton.ConstexprFloat

inductive Expr where
  | natural (value : Nat)
  | literal (value : ℝ)
  | neg (value : Expr)
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  | div (left right : Expr)
  | pow (left right : Expr)

/-- Primitive compile-time operations, rather than a callback for an entire
expression or kernel. A floating interpretation supplies its own rounding. -/
structure Model where
  natural : Nat → ℝ
  literal : ℝ → ℝ
  neg : ℝ → ℝ
  add : ℝ → ℝ → ℝ
  sub : ℝ → ℝ → ℝ
  mul : ℝ → ℝ → ℝ
  div : ℝ → ℝ → ℝ
  pow : ℝ → ℝ → ℝ

def eval (model : Model) : Expr → ℝ
  | .natural n=>model.natural n
  | .literal x=>model.literal x
  | .neg a=>model.neg (eval model a)
  | .add a b=>model.add (eval model a) (eval model b)
  | .sub a b=>model.sub (eval model a) (eval model b)
  | .mul a b=>model.mul (eval model a) (eval model b)
  | .div a b=>model.div (eval model a) (eval model b)
  | .pow a b=>model.pow (eval model a) (eval model b)

noncomputable def real : Model :=
  ⟨Nat.cast,id,Neg.neg,HAdd.hAdd,HSub.hSub,HMul.hMul,HDiv.hDiv,Real.rpow⟩
end VeriTile.Triton.ConstexprFloat

namespace VeriTile.Triton.DSL.ConstexprFloatExpressions
open Lean Elab Term Meta

syntax (name := withModel) "with_constexpr_float_model " term " in " term : term
private def modelOption : Name := `veritile.constexprFloatModel

private def freshName (stem : String) : TermElabM Name := do
  repeat
    let fresh ← mkFreshId
    let name:=Name.mkSimple (stem++(fresh.toString.map fun c=>if c.isAlphanum then c else '_'))
    if ((← getLCtx).findFromUserName? name).isNone then return name
  throwError "could not allocate a hygienic constexpr binding"

@[term_elab withModel]
def elaborateModel : TermElab := fun stx expected => do
  let model ← elabTermEnsuringType stx[1] (mkConst ``ConstexprFloat.Model)
  let name ← freshName "__veritile_constexpr_model_"
  withLetDecl name (mkConst ``ConstexprFloat.Model) model fun localModel => do
    let result ← withOptions (fun opts=>opts.set modelOption name.toString) <|
      elabTerm stx[3] expected
    mkLetFVars #[localModel] result

private partial def listValues (value : Lean.Expr) : TermElabM (List Lean.Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    let args:=value.getAppArgs
    return args[1]!::(← listValues args[2]!)
  throwError "constexpr argument names must be a static list"

private def stringValue (value : Lean.Expr) : TermElabM String := do
  let .lit (.strVal name) ← whnf value | throwError "constexpr argument name must be a string"
  return name

private structure Constant where
  tree : TSyntax `term
  floating : Bool
  power : Bool

private partial def constant? (names : List Name) (raw : TSyntax `tritonExpr) : TermElabM (Option Constant) := do
  match raw with
  | `(tritonExpr| $n:num)=>return some ⟨← `(ConstexprFloat.Expr.natural $n),Bool.false,Bool.false⟩
  | `(tritonExpr| $n:scientific)=>return some ⟨← `(ConstexprFloat.Expr.literal ($n : ℝ)),Bool.true,Bool.false⟩
  | `(tritonExpr| $id:ident)=>
    unless names.contains id.getId.eraseMacroScopes do return none
    let value ← elabTerm id.raw none
    unless ← isDefEq (← inferType value) (mkConst ``Nat) do
      return none
    return some ⟨← `(ConstexprFloat.Expr.natural $id),Bool.false,Bool.false⟩
  | `(tritonExpr| ($e:tritonExpr))=>constant? names e
  | `(tritonExpr| -$e:tritonExpr)=>
    let some e ← constant? names e | return none
    return some ⟨← `((ConstexprFloat.Expr.neg ($e.tree))),e.floating,e.power⟩
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr ** $b:tritonExpr)=>
    let some x ← constant? names a | return none
    let some y ← constant? names b | return none
    let (tree,isPower) ← match raw with
      | `(tritonExpr| $_:tritonExpr + $_:tritonExpr)=>pure (← `((ConstexprFloat.Expr.add ($x.tree) ($y.tree))),Bool.false)
      | `(tritonExpr| $_:tritonExpr - $_:tritonExpr)=>pure (← `((ConstexprFloat.Expr.sub ($x.tree) ($y.tree))),Bool.false)
      | `(tritonExpr| $_:tritonExpr * $_:tritonExpr)=>pure (← `((ConstexprFloat.Expr.mul ($x.tree) ($y.tree))),Bool.false)
      | `(tritonExpr| $_:tritonExpr / $_:tritonExpr)=>pure (← `((ConstexprFloat.Expr.div ($x.tree) ($y.tree))),Bool.false)
      | _=>pure (← `((ConstexprFloat.Expr.pow ($x.tree) ($y.tree))),Bool.true)
    let floatingDivision:=match raw with
      | `(tritonExpr| $_:tritonExpr / $_:tritonExpr)=>Bool.true | _=>Bool.false
    if isPower && !x.floating && !y.floating then
      throwErrorAt raw "integer constexpr power is outside the floating-expression subset"
    if !isPower && !floatingDivision && !x.floating && !y.floating then return none
    return some ⟨tree,x.floating || y.floating || isPower || floatingDivision,x.power || y.power || isPower⟩
  | _=>return none

private partial def lower (names : List Name) (model : Ident) (raw : Syntax) :
    TermElabM (Syntax × List (Name × Lean.Expr)) := do
  if raw.getKind==choiceKind then return ← lower names model raw[0]!
  if let some value ← constant? names ⟨raw⟩ then
    if value.power then
      let tree ← elabTermEnsuringType value.tree (mkConst ``ConstexprFloat.Expr)
      let interpreted ← mkAppM ``ConstexprFloat.eval #[← elabTerm model none,tree]
      let name ← freshName "__veritile_constexpr_value_"
      let id:=mkIdent name
      return ((← `(tritonExpr| $(($id : ℝ)))).raw,[(name,interpreted)])
  if let `(tritonExpr| $_:tritonExpr ** $_:tritonExpr) := (⟨raw⟩ : TSyntax `tritonExpr) then
    throwErrorAt raw "floating power requires an immutable declared constexpr expression"
  match raw with
  | .node info kind args=>
    let lowered ← args.mapM (lower names model)
    return (.node info kind (lowered.map Prod.fst),lowered.toList.flatMap Prod.snd)
  | other=>return (other,[])

private def bindConstants (bindings : List (Name × Lean.Expr))
    (body : TermElabM Lean.Expr) : TermElabM Lean.Expr := do
  let rec bind (bindings : List (Name × Lean.Expr)) : TermElabM Lean.Expr := do
    match bindings with
    | []=>body
    | (name,value)::tail=>
      withLetDecl name (mkConst ``Real) value fun bound=>do
        let result ← bind tail
        mkLetFVars #[bound] result
  bind bindings

private def prepare (stx : Syntax) : TermElabM (Syntax × List (Name × Lean.Expr)) := do
  let modelName:=((← getOptions).get modelOption "").toName
  if modelName.isAnonymous then throwUnsupportedSyntax
  let model:=mkIdent modelName
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let names ← (← listValues (← mkAppM ``CallSignature.constexpr #[profile])).mapM stringValue
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok text=>pure text | .error message=>throwErrorAt raw message
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw message
  let `(triton { $body:tritonStmt* }) := parsed | throwError "expected a complete constexpr call body"
  let lowered ← body.mapM fun statement=>lower (names.map Name.mkSimple) model statement.raw
  let source:=String.intercalate "\n" (lowered.toList.map (fun p=>PythonFullConstructors.render p.1))
  let next:=Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  return (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) next),lowered.toList.flatMap Prod.snd)

@[term_elab CallSignature.withCall]
def elaborateCall : TermElab := fun stx expected => do
  let (stx,bindings) ← prepare stx
  bindConstants bindings (IntegerWhereLiterals.elaborateInvocation stx expected)

@[term_elab CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  let (stx,bindings) ← prepare stx
  bindConstants bindings (PythonInlineSuites.elaborateSignature stx expected)
end VeriTile.Triton.DSL.ConstexprFloatExpressions
