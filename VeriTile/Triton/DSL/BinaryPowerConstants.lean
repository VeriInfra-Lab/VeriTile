import VeriTile.Triton.DSL.PythonInlineSuites

/-! Exact floating constexpr powers. Nonnegative powers of two through 2^127
are exactly representable in Python binary64 and in the selected FP32 literal
channel. They are compile-time values, never runtime pow instructions. Other
floating constexpr expressions remain unsupported by this optional frontend. -/
namespace VeriTile.Triton.DSL.BinaryPowerConstants
open Lean Elab Term Meta

syntax:75 tritonExpr:76 " ** " tritonExpr:75 : tritonExpr

private partial def closedInteger (value : TSyntax `tritonExpr) : TermElabM Int := do
  match value with
  | `(tritonExpr| $n:num) =>
    if n.getNat > 1000000 then throwErrorAt value "constexpr integer is too large"
    return Int.ofNat n.getNat
  | `(tritonExpr| ($e:tritonExpr)) => closedInteger e
  | `(tritonExpr| -$e:tritonExpr) => return -(← closedInteger e)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return (← closedInteger a)+(← closedInteger b)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => return (← closedInteger a)-(← closedInteger b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => return (← closedInteger a)*(← closedInteger b)
  | _ => throwErrorAt value "binary-power exponent must be a closed integer expression"

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      for id in args.take index do
        if id.isIdent && names.contains id.getId.eraseMacroScopes then
          throwErrorAt id "a floating constexpr cannot be reassigned"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "a loop cannot shadow a floating constexpr"
  for arg in raw.getArgs do rejectWrites names arg

private partial def substitute (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_, value) := values.find? (fun p => p.1==id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| $(($value:ident : ℝ)))).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute values))
  | other => return other

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

private def elaborate (stx : Syntax) (signatureOnly : Bool) (expected : Option Expr) : TermElabM Expr := do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete constexpr body"
  let mut body : List (TSyntax `tritonStmt) := []
  let mut values : List (Name × Expr) := []
  for statement in statements do
    if let `(tritonStmt| $id:ident : tl.constexpr = 2.0 ** $exponent:tritonExpr) := statement then
      let name := id.getId.eraseMacroScopes
      if values.any (fun p => p.1==name) then throwErrorAt id "duplicate floating constexpr"
      -- Earlier uses cannot silently become references to the later constant.
      if body.any (fun s => (identifiers s.raw).contains name) then
        throwErrorAt id "floating constexpr used before its binding"
      let n ← closedInteger exponent
      unless 0≤n && n≤127 do throwErrorAt exponent "exact FP32 binary-power constexpr requires exponent in [0,127]"
      let number : TSyntax `num := ⟨Syntax.mkNumLit (toString n.toNat)⟩
      let value ← elabTerm (← `(term| ((2 : ℝ) ^ ($number:num : Nat)))) none
      values:=values++[(name,value)]
    else
      rejectWrites (values.map Prod.fst) statement.raw
      body:=body++[statement]
  if values.isEmpty then
    if signatureOnly then return ← PythonInlineSuites.elaborateSignature stx expected
    else return ← PythonInlineSuites.elaborateInvocation stx expected
  let used := identifiers parsed ++ identifiers stx
  let rec bind (todo : List (Name × Expr)) (boundValues : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName:=Name.mkSimple ("veritile_binary_power_"++(fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains localName do break
      withLetDecl localName (mkConst ``Real) value fun bound => do
        let result ← bind tail (boundValues++[(name,mkIdent localName)])
        mkLetFVars #[bound] result
    | [] =>
      let lowered ← body.mapM (fun s => liftMacroM <| substitute boundValues s.raw)
      let raw := Syntax.atom .none ("{\n"++String.intercalate "\n" (lowered.map render)++"\n}")
      let stx := stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
      if signatureOnly then PythonInlineSuites.elaborateSignature stx expected
      else PythonInlineSuites.elaborateInvocation stx expected
  bind values []

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => elaborate stx Bool.true expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => elaborate stx Bool.false expected

end VeriTile.Triton.DSL.BinaryPowerConstants
