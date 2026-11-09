import VeriTile.Triton.DSL.PythonFullConstructors

/-! Top-level compile-time assertions over immutable integer/Boolean caller
metadata. Evaluate their predicates as host integer expressions, outside the
floating algebra. A rejected assertion has no successful execution and is
unsafe in the model; a satisfied assertion preserves the complete state.
Nested assertions and predicates depending on runtime values are refused.
This optional frontend does not change other quotations or the shared core. -/
namespace VeriTile.Triton.DSL.StaticAssertions
open Lean Elab Term Meta

/-- Compilation predicates are uniform caller metadata. Failed checks have
no executable program; satisfied checks preserve the whole compiled body. -/
def checked (condition : Bool) : ComputeKernel → ComputeKernel
  | .mk inputs outputs code =>
    .mk inputs outputs
      (.ifThen (.constBool (!condition)) [.effectMarker "static_assert.rejected"] :: code)

private inductive Kind where | integer | boolean deriving DecidableEq
private abbrev Constants := List (Name × Kind)

private partial def containsAssertion (raw : Syntax) : Bool :=
  match raw with
  | `(tritonStmt| tl.static_assert($_:tritonExpr $[, $_:str]?)) => Bool.true
  | _ => raw.getArgs.any containsAssertion

private partial def writes (raw : Syntax) : List Name :=
  let args := raw.getArgs
  let direct := if args.size ≥ 2 && args[0]!.isIdent && args[1]!.isAtom &&
      ["=", ":=", "+=", "-=", "*=", "/="].contains args[1]!.getAtomVal.trimAscii.toString
    then [args[0]!.getId.eraseMacroScopes] else []
  let binder := if args.size ≥ 2 && args[0]!.isAtom &&
      args[0]!.getAtomVal.trimAscii.toString == "for" && args[1]!.isIdent
    then [args[1]!.getId.eraseMacroScopes] else []
  let tuple := match raw with
    | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) =>
      (#[a,b] ++ more).toList.map (·.getId.eraseMacroScopes)
    | _ => []
  direct ++ binder ++ tuple ++ args.toList.flatMap writes

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "static assertion metadata must have literal names"

private def constants (profile : Syntax) (changed : List Name) : TermElabM Constants := do
  let profile ← elabTermEnsuringType profile (mkConst ``CallSignature)
  let mut result := []
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr name must be literal"
    unless changed.contains name.toName do
      let value ← elabTerm (mkIdent name.toName) none
      let ty ← whnf (← inferType value)
      if ty.isConstOf ``Nat || ty.isConstOf ``Int then
        result := (name.toName, Kind.integer) :: result
      else if ty.isConstOf ``Bool then
        result := (name.toName, Kind.boolean) :: result
  return result

private def requireKind (actual expected : Kind) : TermElabM Unit :=
  unless actual == expected do throwError "static assertion predicate has incompatible host types"

private partial def host (known : Constants) (e : TSyntax `tritonExpr) :
    TermElabM (TSyntax `term × Kind) := do
  match e with
  | `(tritonExpr| $n:num) => return (← `(term| (Int.ofNat $n:num)), .integer)
  | `(tritonExpr| ($e:tritonExpr)) => host known e
  | `(tritonExpr| $id:ident) =>
    let name := id.getId.eraseMacroScopes
    if ["True", "true", "Bool.true"].contains name.toString then return (← `(term| Bool.true), .boolean)
    if ["False", "false", "Bool.false"].contains name.toString then return (← `(term| Bool.false), .boolean)
    let some kind := known.lookup name
      | throwError "static assertions may only read immutable integer/Boolean constexpr arguments"
    let ty ← whnf (← inferType (← elabTerm id none))
    if ty.isConstOf ``Nat then return (← `(term| (Int.ofNat $id:ident)), kind)
    return (⟨id.raw⟩, kind)
  | `(tritonExpr| -$a:tritonExpr) =>
    let (a,k) ← host known a; requireKind k .integer
    return (← `(term| (-$a:term)), .integer)
  | `(tritonExpr| not $a:tritonExpr) =>
    let (a,k) ← host known a; requireKind k .boolean
    return (← `(term| (!$a:term)), .boolean)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
    let (left,lk) ← host known a; let (right,rk) ← host known b
    requireKind lk .integer; requireKind rk .integer
    let value ← match e with
      | `(tritonExpr| $_:tritonExpr + $_:tritonExpr) => `(term| ($left:term + $right:term))
      | `(tritonExpr| $_:tritonExpr - $_:tritonExpr) => `(term| ($left:term - $right:term))
      | _ => `(term| ($left:term * $right:term))
    return (value, .integer)
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) =>
    let (left,lk) ← host known a; let (right,rk) ← host known b
    requireKind lk rk
    let predicate ← match e with
      | `(tritonExpr| $_:tritonExpr == $_:tritonExpr) => `(term| $left:term = $right:term)
      | `(tritonExpr| $_:tritonExpr != $_:tritonExpr) => `(term| $left:term ≠ $right:term)
      | `(tritonExpr| $_:tritonExpr < $_:tritonExpr) =>
        requireKind lk .integer; `(term| $left:term < $right:term)
      | `(tritonExpr| $_:tritonExpr <= $_:tritonExpr) =>
        requireKind lk .integer; `(term| $left:term ≤ $right:term)
      | `(tritonExpr| $_:tritonExpr > $_:tritonExpr) =>
        requireKind lk .integer; `(term| $left:term > $right:term)
      | _ => requireKind lk .integer; `(term| $left:term ≥ $right:term)
    return (← `(term| (decide ($predicate:term) : Bool)), .boolean)
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) =>
    let (left,lk) ← host known a; let (right,rk) ← host known b
    requireKind lk .boolean; requireKind rk .boolean
    let value ← match e with
      | `(tritonExpr| $_:tritonExpr and $_:tritonExpr) => `(term| ($left:term && $right:term))
      | _ => `(term| ($left:term || $right:term))
    return (value, .boolean)
  | _ => throwError "unsupported static assertion predicate"

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok value => pure value | .error message => throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}") with
    | .ok value => pure value | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete caller body"
  if !statements.any (containsAssertion ·.raw) then
    return ← CallSignature.elaborateInvocation stx expected
  let known ← constants stx[1] (statements.toList.flatMap (writes ·.raw))
  let mut lowered := #[]
  let mut accepted : TSyntax `term ← `(term| Bool.true)
  for statement in statements do
    match statement with
    | `(tritonStmt| tl.static_assert($condition:tritonExpr $[, $_:str]?)) =>
      let (guard,kind) ← host known condition
      requireKind kind .boolean
      accepted ← `(term| ($accepted:term && $guard:term))
    | _ =>
      if containsAssertion statement.raw then throwError "static assertions must be top-level statements"
      lowered := lowered.push statement
  let raw := Syntax.atom raw.getHeadInfo
    ("{\n" ++ String.intercalate "\n" (lowered.toList.map (PythonFullConstructors.render ·.raw)) ++ "\n}")
  let body ← CallSignature.elaborateInvocation
    (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
  -- Compilation checks precede runtime execution. Keep the compiled body
  -- intact and preserve all ports; a failed predicate selects an explicit
  -- rejected-compilation marker, which both execution and safety reject.
  let predicate ← elabTermEnsuringType accepted (mkConst ``Bool)
  mkAppM ``checked #[predicate,body]
end VeriTile.Triton.DSL.StaticAssertions
