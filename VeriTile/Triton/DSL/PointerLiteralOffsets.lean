import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.StaticRanges

/-! A Python integer added to a pointer is an address offset, including when
that pointer is a local register. Give only those literals the natural address
carrier internally; leave scalar arithmetic and the pasted source untouched.
Pointer facts follow sequential assignments and never escape a control arm. -/
namespace VeriTile.Triton.DSL.PointerLiteralOffsets
open Lean Elab Term Meta

private partial def pointer (aliases : List Name) (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $id:ident) => aliases.contains id.getId.eraseMacroScopes
  | `(tritonExpr| ($value:tritonExpr)) => pointer aliases value
  | `(tritonExpr| $base:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $base:tritonExpr - $_:tritonExpr) => pointer aliases base
  | _ => Bool.false

private partial def offset (value : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  match value with
  | `(tritonExpr| $n:num) => `(tritonExpr| $($n:num))
  | `(tritonExpr| ($value:tritonExpr)) => offset value
  | _ => pure value

private partial def choose (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then choose raw[0]!
  else raw.setArgs (raw.getArgs.map choose)

private partial def expression (aliases : List Name) (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM (expression aliases))
  if let `(tritonExpr| $left:tritonExpr + $right:tritonExpr) := raw then
    if pointer aliases left then
      let right ← offset right
      return (← `(tritonExpr| $left + $right)).raw
    if pointer aliases right then
      let left ← offset left
      return (← `(tritonExpr| $left + $right)).raw
  return raw

/-- No pointer fact introduced in a branch or loop escapes that scope. The
base call elaborator also rejects incompatible dynamic pointer rebindings. -/
partial def lower (aliases : List Name) (stmts : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let mut aliases := aliases
  let mut result := #[]
  for original in stmts do
    let stmt : TSyntax `tritonStmt := ⟨choose original.raw⟩
    let key := match stmt with
      | `(tritonStmt| $id:ident = $_:tritonExpr) => some id.getId.eraseMacroScopes
      | _ => none
    let value := match stmt with
      | `(tritonStmt| $_:ident = $value:tritonExpr) => some value
      | _ => none
    let normalized ← match stmt with
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => do
          let yes ← lower aliases yes; let no ← lower aliases no
          `(tritonStmt| if $c { $yes* } else { $no* })
      | `(tritonStmt| if $c:tritonExpr { $yes:tritonStmt* }) => do
          let yes ← lower aliases yes
          `(tritonStmt| if $c { $yes* })
      | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr) { $inner:tritonStmt* }) => do
          let inner ← lower (aliases.filter (· != i.getId.eraseMacroScopes)) inner
          `(tritonStmt| for $i in range(0, $stop) { $inner* })
      | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr, $step:tritonExpr) { $inner:tritonStmt* }) => do
          let inner ← lower (aliases.filter (· != i.getId.eraseMacroScopes)) inner
          `(tritonStmt| for $i in range(0, $stop, $step) { $inner* })
      | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr) { $inner:tritonStmt* }) => do
          let inner ← lower (aliases.filter (· != i.getId.eraseMacroScopes)) inner
          `(tritonStmt| for $i in range($a, $b) { $inner* })
      | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $step:tritonExpr) { $inner:tritonStmt* }) => do
          let inner ← lower (aliases.filter (· != i.getId.eraseMacroScopes)) inner
          `(tritonStmt| for $i in range($a, $b, $step) { $inner* })
      | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $inner:tritonStmt* }) => do
          let inner ← lower (aliases.filter (· != i.getId.eraseMacroScopes)) inner
          `(tritonStmt| for $i in range($stop) { $inner* })
      | _ => pure (⟨← expression aliases stmt.raw⟩ : TSyntax `tritonStmt)
    result := result.push normalized
    if let some key := key then
      let isPointer := value.any (pointer aliases)
      aliases := aliases.filter (· != key)
      if isPointer then aliases := key :: aliases
  return result

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "pointer parameters must form a static list"

private def prepare (stx : Syntax) : TermElabM Syntax := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut aliases := []
  for projection in [``Triton.CallSignature.pointers, ``Triton.CallSignature.floatingPointers] do
    for item in ← listValues (← mkAppM projection #[profile]) do
      let name ← whnf (← mkAppM ``Prod.fst #[item])
      let .lit (.strVal name) := name | throwError "pointer names must be literal"
      aliases := Name.mkSimple name :: aliases
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parserEnv := Parser.parserExtension.activateScoped (← getEnv)
    `VeriTile.Triton.DSL.StaticRanges
  let parsed ← match Parser.runParserCategory parserEnv `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "complete source body required"
  let stmts ← liftMacroM (lower aliases stmts)
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  StaticRanges.elaborateInvocation (← prepare stx) expected
end VeriTile.Triton.DSL.PointerLiteralOffsets
