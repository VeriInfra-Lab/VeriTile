import VeriTile.Triton.DSL.OutputDTypeMetadata

/-! Optional support for Python scalar infinity in a narrow floating where.
Triton v3.7.1 keeps the tensor dtype when the scalar has the same kind:
https://github.com/triton-lang/triton/blob/v3.7.1/docs/python-api/triton-semantics.rst
Track only established floating/pointer types, forget facts across control
flow, and reject an infinity select whose tensor dtype cannot be established.
The implicit conversion is emitted inside elaboration, never in pasted source. -/
namespace VeriTile.Triton.DSL.ScalarInfinityWhere
open Lean Elab Term Meta

syntax (name := withInfinityWhere) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " scalar_infinity_where " "in " "triton " RawSourceText.body : term

private structure Types where
  pointers : List (Name × FloatDType) := []
  floats : List (Name × FloatDType) := []
  constants : List (Name × FloatDType) := []

private def dtype? (types : Types) (dt : TSyntax `tritonDType) : Option FloatDType :=
  match dt with
  | `(tritonDType| tl.float16) => some .fp16
  | `(tritonDType| tl.bfloat16) => some .bf16
  | `(tritonDType| tl.float32) => some .fp32
  | `(tritonDType| tl.float64) => some .real
  | `(tritonDType| OUT_DTYPE) => (types.constants.find? (·.1 == `OUT_DTYPE)).map Prod.snd
  | _ => none

private partial def infinity (raw : Syntax) : Bool :=
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| float($s:str)) => ["inf","+inf","-inf"].contains s.getString
  | `(tritonExpr| -$e:tritonExpr) | `(tritonExpr| ($e:tritonExpr)) => infinity e
  | _ => Bool.false

private partial def scalar (raw : Syntax) : Bool :=
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $_:num) => Bool.true
  | `(tritonExpr| -$e:tritonExpr) | `(tritonExpr| ($e:tritonExpr)) => scalar e
  | _ => infinity raw

private partial def pointer? (types : Types) (raw : Syntax) : Option FloatDType :=
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $id:ident) => (types.pointers.find? (·.1 == id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| $a:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $_:tritonExpr)
  | `(tritonExpr| ($a:tritonExpr)) => pointer? types a
  | `(tritonExpr| tl.make_block_ptr($bk:ident=$base:tritonExpr,
      $sk:ident=($_:tritonExpr,*), $stk:ident=($_:tritonExpr,*),
      $ok:ident=($_:tritonExpr,*), $bs:ident=($_:tritonExpr,*), $ord:ident=($_:num,*))) =>
    if [bk,sk,stk,ok,bs,ord].map (·.getId.eraseMacroScopes.toString) ==
        ["base","shape","strides","offsets","block_shape","order"] then pointer? types base else none
  | _ => none

private def methodCastType? (types : Types) (raw : Syntax) : Option FloatDType := do
  unless [``tritonMethodCast, ``tritonIdentMethodCast, ``tritonIdentMethodCastSpaced,
      ``tritonDottedIdentMethodCast].contains raw.getKind do none
  let args := raw.getArgs
  let opening ← args.findIdx? (fun token => token.isAtom && token.getAtomVal == "(")
  let target ← args[opening + 1]?
  dtype? types ⟨target⟩

private partial def floating? (types : Types) (raw : Syntax) : Option FloatDType := do
  if let some dtype := methodCastType? types raw then return dtype
  let combine (a b : Syntax) : Option FloatDType := do
    if scalar a then return ← floating? types b
    if scalar b then return ← floating? types a
    let a ← floating? types a
    let b ← floating? types b
    if a == b then return a
    if a == .real || b == .real then return .real
    if a == .fp32 || b == .fp32 then return .fp32
    if (a == .fp16 && b == .bf16) || (a == .bf16 && b == .fp16) then return .fp16
    none
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $id:ident) => (types.floats.find? (·.1 == id.getId.eraseMacroScopes)).map Prod.snd
  | `(tritonExpr| ($a:tritonExpr)) => floating? types a
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*], dtype=$dt:tritonDType)) => dtype? types dt
  | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) => pointer? types p
  | `(tritonExpr| ($v:tritonExpr).to($dt:tritonDType)) => dtype? types dt
  | `(tritonExpr| tl.cast($v:tritonExpr, $dt:tritonDType)) => dtype? types dt
  | `(tritonExpr| $callee:ident($dt:tritonDType)) =>
    if callee.getId.eraseMacroScopes.toString.endsWith ".to" then dtype? types dt else none
  | `(tritonExpr| tl.dot($_:tritonExpr, $_:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
    let selected := kwargs.toList.filterMap fun kw => match kw with
      | `(tritonReduceKwarg| out_dtype=$dt:tritonDType) => dtype? types dt
      | _ => none
    selected.head?.orElse (fun _ => some .fp32)
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => combine a b
  | `(tritonExpr| tl.where($_:tritonExpr, $a:tritonExpr, $b:tritonExpr)) => combine a b
  | _ => none

private def dtypeSyntax (dtype : FloatDType) : MacroM (TSyntax `tritonDType) :=
  match dtype with
  | .fp16 => `(tritonDType| tl.float16)
  | .bf16 => `(tritonDType| tl.bfloat16)
  | .fp32 => `(tritonDType| tl.float32)
  | .real => `(tritonDType| tl.float64)
  | _ => Macro.throwError "unsupported scalar infinity tensor dtype"

private partial def expression (types : Types) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← expression types raw[0]!
  let raw := raw.setArgs (← raw.getArgs.mapM (expression types))
  if let `(tritonExpr| tl.where($condition:tritonExpr, $yes:tritonExpr, $no:tritonExpr)) := raw then
    let isYes := infinity yes
    let isNo := infinity no
    if isYes != isNo then
      let tensor := if isYes then no else yes
      let some dtype := floating? types tensor
        | Macro.throwErrorAt raw "scalar infinity where requires a statically known floating tensor branch"
      if dtype == .fp16 || dtype == .bf16 then
        let dt ← dtypeSyntax dtype
        let weak := if isYes then yes else no
        let converted ← `(tritonExpr| ($weak:tritonExpr).to($dt:tritonDType))
        if isYes then return (← `(tritonExpr| tl.where($condition, $converted, $no))).raw
        else return (← `(tritonExpr| tl.where($condition, $yes, $converted))).raw
  return raw

private def put (entries : List (Name × FloatDType)) (name : Name) (value : Option FloatDType) :=
  entries.filter (·.1 != name) ++ value.toList.map (name, ·)

private partial def statements (types : Types) (body : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let mut types := types
  let mut out := #[]
  for st in body do
    if let `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $loop:tritonStmt* }) := st then
      let loop ← statements {types with
        pointers := types.pointers.filter (·.1 != i.getId.eraseMacroScopes)
        floats := types.floats.filter (·.1 != i.getId.eraseMacroScopes)} loop
      out := out.push (← `(tritonStmt| for $i:ident in range($start, $stop, $stride) { $loop* }))
      types := {types with pointers := [], floats := []}
      continue
    if let some first := st.raw.getArgs[0]? then
      if first.isAtom && ["if","for","while"].contains first.getAtomVal.trimAscii.toString then
        Macro.throwErrorAt st "scalar infinity typing requires straight-line statements or a three-argument range"
    let lowered : TSyntax `tritonStmt := ⟨← expression types st⟩
    if let `(tritonStmt| $id:ident = $value:tritonExpr) := lowered then
      let name := id.getId.eraseMacroScopes
      types := {types with
        pointers := put types.pointers name (pointer? types value)
        floats := put types.floats name (floating? types value)}
    else if let `(tritonStmt| $id:ident += $value:tritonExpr) := lowered then
      let name := id.getId.eraseMacroScopes
      let whole ← `(tritonExpr| $id:ident + $value:tritonExpr)
      types := {types with floats := put types.floats name (floating? types whole)}
    else if let `(tritonStmt| $id:ident -= $_:tritonExpr) := lowered then
      types := {types with floats := put types.floats id.getId.eraseMacroScopes none}
    else
      -- Unhandled control and assignments cannot preserve stale typing facts.
      match lowered with
      | `(tritonStmt| tl.store($_p:tritonExpr, $_v:tritonExpr)) => pure ()
      | `(tritonStmt| tl.store($_p:tritonExpr, $_v:tritonExpr, $_kw:tritonMemKwarg $[, $_more:tritonMemKwarg]*)) => pure ()
      | _ => types := {types with pointers := [], floats := []}
    out := out.push lowered
  return out

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "scalar infinity metadata requires literal pointer bindings"

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind == choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out := out ++ " "
    out := out ++ token
    previous := token
  return out

@[term_elab withInfinityWhere] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut types : Types := {}
  for entry in ← listValues (← mkAppM ``CallSignature.pointers #[profile]) do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry]) | throwError "pointer names must be literal"
    let kind ← whnf (← mkAppM ``Prod.snd #[entry])
    let dtype := if kind.isConstOf ``SourceDType.fp16 then some FloatDType.fp16
      else if kind.isConstOf ``SourceDType.bf16 then some FloatDType.bf16
      else if kind.isConstOf ``SourceDType.fp32 then some FloatDType.fp32 else none
    types := {types with pointers := put types.pointers name.toName dtype}
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr names must be literal"
    if name == "OUT_DTYPE" then
      let value ← elabTerm (mkIdent name.toName) (some (mkConst ``FloatDType))
      let value ← whnf value
      if value.isConstOf ``FloatDType.fp16 then types := {types with constants := [(name.toName,.fp16)]}
      else if value.isConstOf ``FloatDType.bf16 then types := {types with constants := [(name.toName,.bf16)]}
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}") with
    | .ok parsed => pure parsed | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let lowered ← liftMacroM <| statements types stmts
  let raw := Syntax.atom .none ("{\n" ++ String.intercalate "\n" (lowered.toList.map (fun s => render s.raw)) ++ "\n}")
  let args := #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[7],stx[8],raw]
  OutputDTypeMetadata.elaborateInvocation (mkNode ``CallSignature.withCall args) expected
end VeriTile.Triton.DSL.ScalarInfinityWhere
