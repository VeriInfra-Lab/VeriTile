import VeriTile.Triton.DSL.CallValue
import VeriTile.Triton.DSL.SourceBindings
import VeriTile.Triton.DSL.PythonConstants
import VeriTile.Triton.DSL.SourceDType
import VeriTile.Triton.DSL.RawBodyControl
import VeriTile.Triton.DSL.RawSourceText
import VeriTile.Triton.DSL.RawBodyConstants
import VeriTile.Triton.DSL.PositionalBlockPointer
import VeriTile.Triton.DSL.RawBodyStores
import VeriTile.Triton.DSL.RawBodyDTypeControl
import VeriTile.Triton.DSL.RawBodyNestedLoads
import Lean
import Lean.Meta.Tactic.Cases

/-! Call-site typing for a pasted Triton body. The signature describes pointer
storage and runtime scalar registers; it emits no kernel instruction. The body
uses ordinary Triton names, casts and implicit destination conversions. -/
namespace VeriTile.Triton

/-- Scalar carrier/precision supplied by the invocation. `integer` is the
mathematical integer channel, not a claim about an unannotated Python ABI width. -/
inductive ScalarType where
  | fp32 | fp64 | integer | i32 | i64

structure CallSignature where
  pointers : List (String × SourceDType) := []
  /-- Floating storage supplied by a polymorphic call site. Unlike the fixed
  source-dtype entries this may depend on the caller's FloatDType parameters. -/
  floatingPointers : List (String × FloatDType) := []
  scalars : List (String × ScalarType) := []
  constexpr : List String := []
  /-- Immutable integer literals from the Python module, checked by source correspondence. -/
  sourceConstants : List (String × String) := []
  /-- Exact Python import aliases for the supported libdevice unary intrinsic. -/
  unaryImports : List (String × String) := []
  /-- Explicit invocation port order for kernels whose first access order differs. -/
  readPorts : List String := []
  writePorts : List String := []

/-- A scalar expression supplied by a call, before the callee's first statement.
These bindings are separate from the pasted source body and preserve dtype. -/
structure ScalarArgument where
  name : RegName
  dtype : TileDType
  value : ComputeExpr dtype []

namespace ScalarArgument
noncomputable def fp32 (name : RegName) (value : ℝ) : ScalarArgument :=
  ⟨name,.real,.compute (.alg .fp32 (.const value))⟩
def integer (name : RegName) (value : Int) : ScalarArgument :=
  ⟨name,.int,.alg (.constInt value)⟩
def statement (arg : ScalarArgument) : ComputeStmt :=
  .assign arg.dtype [] arg.name arg.value
end ScalarArgument

/-- Instantiate a call without changing its source quotation. The prefix only
binds argument expressions, in order; it performs no memory operation. -/
def ComputeKernel.withScalarArguments (kernel : ComputeKernel) (args : List ScalarArgument) : ComputeKernel :=
  .mk kernel.inputs kernel.outputs (args.map ScalarArgument.statement ++ kernel.surfaceBody)


namespace DSL.CallSignature
open Lean Elab Term Meta

syntax "tl.zeros(" "(" tritonExpr,* ")" ")" : tritonExpr
syntax "tl.zeros(" "(" tritonExpr,* ")" ", " tritonDType ")" : tritonExpr
syntax "tl.zeros(" "(" tritonExpr,* ")" ", " ident "=" tritonDType ")" : tritonExpr
syntax "for " ident " in " "range(0, " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax (priority := low) "for " ident " in " "range(" tritonExpr ", " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax (priority := low) "for " ident " in " "range(0, " tritonExpr ", " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax ident ", " ident (", " ident)* " = " "(" tritonExpr ", " tritonExpr (", " tritonExpr)* ")" : tritonStmt

private partial def pythonIntegerLiteral? (value : TSyntax `tritonExpr) : Option Int :=
  match value with
  | `(tritonExpr| $n:num) => some n.getNat
  | `(tritonExpr| ($value:tritonExpr)) => pythonIntegerLiteral? value
  | `(tritonExpr| -$value:tritonExpr) => (pythonIntegerLiteral? value).map Neg.neg
  | _ => none

private def integerDType (dtype : TSyntax `tritonDType) : Bool :=
  match dtype with
  | `(tritonDType| tl.int8) | `(tritonDType| tl.int16) | `(tritonDType| tl.int32)
  | `(tritonDType| tl.int64) | `(tritonDType| tl.uint8) | `(tritonDType| tl.uint16)
  | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64) => Bool.true
  | _ => Bool.false

/-- Keep direct integer/Boolean load casts recognizable by the typed load
lowering before any floating receiver-parenthesization is applied. -/
private def directLoadCast (dtype : TSyntax `tritonDType) : Bool :=
  integerDType dtype || (match dtype with
    | `(tritonDType| tl.int1) => Bool.true
    | _ => Bool.false)

private partial def loadOperand (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| ($value:tritonExpr)) => loadOperand value
  | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
  | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
  | _ => Bool.false

private partial def normalizeRawBody (parserEnv : Environment) (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM (normalizeRawBody parserEnv))
  match raw with
  | `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = ($x:tritonExpr, $y:tritonExpr $[, $rest:tritonExpr]*)) =>
      return (← `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $x:tritonExpr, $y:tritonExpr $[, $rest:tritonExpr]*)).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      if directLoadCast dt then return raw
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      if directLoadCast dt then return raw
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) => do
      unless target.getId.eraseMacroScopes matches .str _ "dtype" do return raw
      let dtype : TSyntax `term := ⟨target.raw⟩
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dtype:term))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) => do
      unless target.getId.eraseMacroScopes matches .str _ "dtype" do return raw
      let dtype : TSyntax `term := ⟨target.raw⟩
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dtype:term))).raw
  | `(tritonExpr| tl.cast($value:tritonExpr, $dtype:tritonDType)) => do
      unless integerDType dtype do return raw
      let some n := pythonIntegerLiteral? value
        | if loadOperand value then
            return (← `(tritonExpr| ($value:tritonExpr).to($dtype:tritonDType))).raw
          else return raw
      unless -(2^63 : Int) ≤ n && n < 2^63 do
        Macro.throwErrorAt value "integer literal cast requires a supported signed-64 source literal"
      let digits : TSyntax `num := ⟨Syntax.mkNumLit (toString n.natAbs)⟩
      let value ← if n < 0 then do
          let term ← `(term| (-($digits:num : Int)))
          `(tritonExpr| $($term))
        else do
          let term ← `(term| ($digits:num : Nat))
          `(tritonExpr| $($term))
      return (← `(tritonExpr| tl.cast($value:tritonExpr, $dtype:tritonDType))).raw
  | `(tritonExpr| $method:ident($target:tritonExpr)) => do
      -- A raw Python token such as x.to is also a legal qualified Lean
      -- identifier. Parenthesize only its receiver to resolve that ambiguity.
      let .str receiver "to" := method.getId.eraseMacroScopes | return raw
      let `(tritonExpr| $name:ident) := target | return raw
      let name := name.getId.eraseMacroScopes
      let isAttribute := match name with
        | .str (.str _ "dtype") "element_ty" | .str (.str _ "type") "element_ty"
        | .str _ "dtype" => Bool.true
        | _ => Bool.false
      unless isAttribute || (← dtypeSyntaxOfName? name).isSome do return raw
      match Parser.runParserCategory parserEnv `tritonExpr
          ("(" ++ receiver.toString ++ ").to(" ++ name.toString ++ ")") with
      | .ok expression => return expression
      | .error message => Macro.throwErrorAt raw message
  | `(tritonExpr| tl.zeros(($dims:tritonExpr,*))) =>
      return (← `(tritonExpr| tl.zeros([$dims,*]))).raw
  | `(tritonExpr| tl.zeros(($dims:tritonExpr,*), $dt:tritonDType)) =>
      return (← `(tritonExpr| tl.zeros([$dims,*], $dt:tritonDType))).raw
  | `(tritonExpr| tl.zeros(($dims:tritonExpr,*), $name:ident=$dt:tritonDType)) =>
      return (← `(tritonExpr| tl.zeros([$dims,*], $name:ident=$dt:tritonDType))).raw
  | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr) { $body:tritonStmt* }) =>
      return (← `(tritonStmt| for $i:ident in range($(0), $stop:tritonExpr, $(1)) { $body* })).raw
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
      return (← `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $(1)) { $body* })).raw
  | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
      return (← `(tritonStmt| for $i:ident in range($(0), $stop:tritonExpr, $step:tritonExpr) { $body* })).raw
  | _ => return raw

declare_syntax_cat tritonCallBinding
syntax ident " := " term : tritonCallBinding
syntax (name:=withCall) "with_triton_call " term " call_args " "[" tritonCallBinding,* "]" " in "
  "triton " RawSourceText.body : term

syntax (name:=withSignature) "with_triton_signature " term " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    let args := value.getAppArgs
    return args[1]! :: (← listValues args[2]!)
  throwError "call signature must be a statically known list"

private def stringValue (value : Expr) : TermElabM String := do
  let value ← whnf value
  if let .lit (.strVal s) := value then return s
  throwError "call signature names must be string literals"

private def dtypeValue (value : Expr) : TermElabM Triton.SourceDType := do
  let value ← whnf value
  let kinds : List (Name × Triton.SourceDType) :=
    [(``Triton.SourceDType.fp16,.fp16),(``Triton.SourceDType.bf16,.bf16),
     (``Triton.SourceDType.fp32,.fp32),(``Triton.SourceDType.fp64,.fp64),
     (``Triton.SourceDType.f8e4,.f8e4),(``Triton.SourceDType.f8e5,.f8e5),
     (``Triton.SourceDType.i8,.i8),(``Triton.SourceDType.i16,.i16),
     (``Triton.SourceDType.i32,.i32),(``Triton.SourceDType.i64,.i64),
     (``Triton.SourceDType.u8,.u8),(``Triton.SourceDType.u16,.u16),
     (``Triton.SourceDType.u32,.u32),(``Triton.SourceDType.u64,.u64),
     (``Triton.SourceDType.boolean,.boolean)]
  if let some (_,kind) := kinds.find? (fun p => value.isConstOf p.1) then return kind
  throwError "call signature dtypes must be statically known"

private def scalarTypeValue (value : Expr) : TermElabM ScalarType := do
  let value ← whnf value
  if value.isConstOf ``ScalarType.fp32 then return .fp32
  if value.isConstOf ``ScalarType.fp64 then return .fp64
  if value.isConstOf ``ScalarType.integer then return .integer
  if value.isConstOf ``ScalarType.i32 then return .i32
  if value.isConstOf ``ScalarType.i64 then return .i64
  throwError "runtime scalar type must be FP32, FP64 or an integer ABI"

private def bindings {α : Type} (decode : Expr → TermElabM α) (value : Expr) :
    TermElabM (List (String × α)) := do
  (← listValues value).mapM fun pair => do
    let pair ← whnf pair
    unless pair.isAppOfArity ``Prod.mk 4 do throwError "invalid call signature binding"
    let args := pair.getAppArgs
    return (← stringValue args[2]!, ← decode args[3]!)

private partial def rejectBodyMetadata (raw : Syntax) : TermElabM Unit := do
  if raw.getKind == ``tritonCallSignature then
    throwErrorAt raw "call-signature metadata belongs outside the pasted body"
  if let `(tritonStmt| tl.region $_:ident = $_:tritonDType $[, $_:ident = $_:tritonDType]*) := raw then
    throwErrorAt raw "pointer types belong in the call signature, outside the pasted body"
  if let `(tritonStmt| tl.param $_:ident : $_:tritonDType) := raw then
    throwErrorAt raw "runtime scalar types belong in the call signature, outside the pasted body"
  for arg in raw.getArgs do rejectBodyMetadata arg

private partial def staticNatBound (constants assigned : List Name)
    (e : TSyntax `tritonExpr) : TermElabM (Option (TSyntax `term)) := do
  match e with
  | `(tritonExpr| ($inner:tritonExpr)) => staticNatBound constants assigned inner
  | `(tritonExpr| $n:num) => return some (← `(($n : Nat)))
  | `(tritonExpr| $($term:term)) =>
      let value ← elabTerm term none
      if ← isDefEq (← inferType value) (mkConst ``Nat) then return some term else return none
  | `(tritonExpr| $id:ident) =>
      let name := id.getId.eraseMacroScopes
      unless constants.contains name && !assigned.contains name do return none
      let term : TSyntax `term := ⟨id.raw⟩
      let value ← elabTerm term none
      if ← isDefEq (← inferType value) (mkConst ``Nat) then return some term else return none
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
      let some a ← staticNatBound constants assigned a | return none
      let some b ← staticNatBound constants assigned b | return none
      return some (← `(term| ((($a:term + $b:term - 1) / $b:term) : Nat)))
  | _ => return none

/-- A per-call size has already been supplied before executing the body.
Select the existing constant-bound range constructor only for known immutable
Nat bounds. Runtime locals and signed casts retain their dynamic range path. -/
private partial def specializeRanges (constants assigned : List Name) (raw : Syntax) : TermElabM Syntax := do
  let args ← raw.getArgs.mapM (specializeRanges constants assigned)
  let next := match raw with | .node info kind _ => Syntax.node info kind args | _ => raw
  if let `(tritonStmt| for $i:ident in range($n:tritonExpr) { $body:tritonStmt* }) := next then
    if let some n ← staticNatBound constants assigned n then
      return (← liftMacroM `(tritonStmt| tl.for $i in $($n) { $body* })).raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr) { $body:tritonStmt* }) := next then
    if let some a ← staticNatBound constants assigned a then
      if let some b ← staticNatBound constants assigned b then
        return (← liftMacroM `(tritonStmt| for $i in range($($a), $($b), $(1)) { $body* })).raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) := next then
    if let some a ← staticNatBound constants assigned a then
      if let some b ← staticNatBound constants assigned b then
        if let some c ← staticNatBound constants assigned c then
          return (← liftMacroM `(tritonStmt| for $i in range($($a), $($b), $($c)) { $body* })).raw
  return next

private partial def readsTupleTarget (target : Name) (raw : Syntax) : Bool :=
  if let `(tritonExpr| $($_:term)) := raw then Bool.false
  else if raw.isIdent then
    let value := raw.getId.eraseMacroScopes.toString
    value == target.toString || value.startsWith (target.toString ++ ".")
  else raw.getArgs.any (readsTupleTarget target)

private partial def checkTupleReads (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $x:tritonExpr, $y:tritonExpr $[, $rest:tritonExpr]*) := raw then
    let lhs := #[a,b] ++ more
    let rhs := #[x,y] ++ rest
    unless lhs.size==rhs.size do Macro.throwErrorAt raw "Python tuple assignment arity differs"
    for i in [:rhs.size] do
      for earlier in lhs.toList.take i do
        if readsTupleTarget earlier.getId.eraseMacroScopes rhs[i]!.raw then
          Macro.throwErrorAt raw "simultaneous tuple reads require an explicit register-capture lowering"
  for arg in raw.getArgs do checkTupleReads arg

private partial def storeValues (raw : Syntax) : List Syntax :=
  match raw with
  | `(tritonStmt| tl.store($_:tritonExpr, $value:tritonExpr)) => [value.raw]
  | `(tritonStmt| tl.store($_:tritonExpr, $value:tritonExpr, $_:tritonMemKwarg $[, $_:tritonMemKwarg]*)) => [value.raw]
  | `(tritonStmt| tl.store($_:tritonExpr, $value:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => [value.raw]
  | _ => raw.getArgs.toList.flatMap storeValues

/-- Immutable runtime integers are supplied by the invocation, not promoted to
Python constexprs. This specializes the mathematical index projection; native
ABI-width bounds remain the caller's explicit contract obligation. -/
private partial def bindSourceValues (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) :
    MacroM Syntax := do
  if raw.isIdent then
    if let some (_, value) := values.find? (fun p => p.1 == raw.getId.eraseMacroScopes) then
      return value.raw
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_, value) := values.find? (fun p => p.1 == id.getId.eraseMacroScopes) then
      return value.raw
  let expression : TSyntax `tritonExpr := ⟨raw⟩
  if let some (receiver, dtype) := methodCast? expression then
    let next ← bindSourceValues values receiver.raw
    if next != receiver.raw then
      let receiver : TSyntax `tritonExpr := ⟨next⟩
      return (← `(tritonExpr| ($receiver).to($dtype:tritonDType))).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (bindSourceValues values))
  | _ => return raw

private partial def rejectArgumentMutation (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let reject (id : Ident) : TermElabM Unit := do
    if names.contains id.getId.eraseMacroScopes then
      throwErrorAt raw "per-call specialization does not yet support reassigned scalar arguments"
  if let `(tritonStmt| $id:ident = $_:tritonExpr) := raw then reject id
  if let `(tritonStmt| $id:ident += $_:tritonExpr) := raw then reject id
  if let `(tritonStmt| $id:ident -= $_:tritonExpr) := raw then reject id
  if let `(tritonStmt| $id:ident *= $_:tritonExpr) := raw then reject id
  if let `(tritonStmt| $id:ident /= $_:tritonExpr) := raw then reject id
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) := raw then
    reject a
    reject b
    for id in more do reject id
  if let `(tritonStmt| $id:ident : tl.constexpr = $_:tritonExpr) := raw then reject id
  if raw.getNumArgs > 1 && raw[0].isAtom && raw[1].isIdent &&
      ["for", "tl.for"].contains raw[0].getAtomVal.trimAscii.toString then
    reject ⟨raw[1]⟩
  for arg in raw.getArgs do rejectArgumentMutation names arg

/-- A top-level self-cast preserves the source argument as a runtime register.
Other mutations still fail closed until their control-flow joins are modeled. -/
private def argumentSignedCast? (raw : Syntax) : Option Name := do
  let `(tritonStmt| $target:ident = $value:tritonExpr) := raw | none
  let (receiver,dtype) ← (methodCast? value) <|> (match value with
    | `(tritonExpr| tl.cast($receiver:tritonExpr, $dtype:tritonDType)) => some (receiver,dtype)
    | _ => none)
  let name ← if receiver.raw.isIdent then some receiver.raw.getId.eraseMacroScopes
    else match receiver with
      | `(tritonExpr| $name:ident) => some name.getId.eraseMacroScopes
      | _ => none
  match dtype with
  | `(tritonDType| tl.int8) | `(tritonDType| tl.int16)
  | `(tritonDType| tl.int32) | `(tritonDType| tl.int64) => pure ()
  | _ => none
  if target.getId.eraseMacroScopes == name
    then some target.getId.eraseMacroScopes else none

/-- Specialize call integers through their checked, pure self-casts. Every
source assignment remains an instruction; later uses receive the cast value,
rather than the original argument. Floating operations are not propagated. -/
private def bindInvocationStatements (values : List (Name × TSyntax `tritonExpr))
    (captured : List Name) (stmts : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let mut current := values
  let mut result := #[]
  for stmt in stmts do
    if let some name := argumentSignedCast? stmt.raw then
      if captured.contains name then
        let `(tritonStmt| $target:ident = $rhs:tritonExpr) := stmt
          | Macro.throwErrorAt stmt "expected an integer argument self-cast"
        let rhs : TSyntax `tritonExpr := ⟨← bindSourceValues current rhs.raw⟩
        result := result.push (← `(tritonStmt| $target:ident = $rhs:tritonExpr))
        current := current.map (fun entry => if entry.1 == name then (name,rhs) else entry)
        continue
    result := result.push ⟨← bindSourceValues current stmt.raw⟩
  return result

/-- Literal specialization cannot erase the source argument's FP32 rounding.
Accept it only when elaboration already placed every use under FP32 evaluation.
Other contexts use the ordinary runtime register interface. -/
private partial def checkFP32CallPrecision (expression : Expr) (fp32 : Bool := Bool.false) : TermElabM Unit := do
  if expression.isAppOf ``CallValue.fp32 then
    unless fp32 do throwError "per-call FP32 literal requires an FP32 computation context; use runtime scalar binding"
    return
  if expression.isAppOf ``ComputeOp.alg then
    let args := expression.getAppArgs
    if args.size ≥ 2 then
      checkFP32CallPrecision args.back! (args[args.size-2]!.isConstOf ``ComputeDType.fp32)
      return
  if expression.isAppOf ``Op.withPrecision then
    let args := expression.getAppArgs
    if args.size ≥ 2 then
      checkFP32CallPrecision args.back! (args[args.size-2]!.isConstOf ``IndexComputePrecision.fp32)
      return
  match expression with
  | .app f a => checkFP32CallPrecision f fp32; checkFP32CallPrecision a fp32
  | .lam _ type body _ | .forallE _ type body _ =>
      checkFP32CallPrecision type fp32
      checkFP32CallPrecision body fp32
  | .letE _ type value body _ =>
      checkFP32CallPrecision type fp32
      checkFP32CallPrecision value fp32
      checkFP32CallPrecision body fp32
  | .mdata _ value | .proj _ _ value => checkFP32CallPrecision value fp32
  | _ => pure ()

/-- A bare function-terminal return falls through the same function end.
Nested and nonterminal returns retain the ordinary control-flow checks. -/
private def withoutTerminalReturn (stmts : Array (TSyntax `tritonStmt)) :
    Array (TSyntax `tritonStmt) :=
  match stmts.back? with
  | some st => if st.raw.getKind == ``tritonReturn then stmts.pop else stmts
  | none => stmts

/-- Specialize only finite dtype discriminators in caller metadata. Source
statements are elaborated unchanged in every branch. Recursive or indexed
inductives and kernel runtime values are never case-split by this helper. -/
private def finitePointerDiscriminator? (profile : Expr) : TermElabM (Option FVarId) := do
  let entries ← bindings pure (← mkAppM ``Triton.CallSignature.pointers #[profile])
  for (_, kind) in entries do
    let kind ← whnf kind
    if kind.isConst then continue
    for id in (collectFVars {} kind).fvarIds do
      let type ← whnf (← inferType (mkFVar id))
      let .const typeName _ := type | continue
      let .inductInfo info ← getConstInfo typeName | continue
      unless info.numParams == 0 && info.numIndices == 0 &&
          !info.ctors.isEmpty && info.ctors.length ≤ 32 do continue
      let finite ← info.ctors.allM fun ctor => do
        let .ctorInfo c ← getConstInfo ctor | return Bool.false
        return c.numFields == 0
      if finite then return some id
  return none

private partial def elaborateCall (profile : TSyntax `term) (raw : Syntax)
    (callValues : Option (Array (TSyntax `tritonCallBinding))) (expected : Option Expr) : TermElabM Expr := do
      let source ← match RawSourceText.normalize raw.getAtomVal with
        | .ok body => pure body
        | .error message => throwErrorAt raw "{message}"
      let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
        | .ok body => pure body
        | .error message => throwErrorAt raw "{message}"
      let `(triton { $stmts:tritonStmt* }) := parsed
        | throwErrorAt raw "expected a complete Triton function body"
      let stmts := withoutTerminalReturn stmts
      let profile ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
      let profile ← instantiateMVars profile
      if let some discriminator ← finitePointerDiscriminator? profile then
        let goal ← mkFreshExprSyntheticOpaqueMVar (mkConst ``Triton.ComputeKernel)
        let branches ← goal.mvarId!.cases discriminator
        for branch in branches do
          branch.mvarId.withContext do
            let selected ← exprToSyntax (branch.subst.apply profile)
            let result ← elaborateCall selected raw callValues (some (← branch.mvarId.getType))
            branch.mvarId.assign result
        return ← instantiateMVars goal
      let pointers ← bindings dtypeValue (← mkAppM ``Triton.CallSignature.pointers #[profile])
      let floating ← bindings pure (← mkAppM ``Triton.CallSignature.floatingPointers #[profile])
      let mut storageBindings : List (Ident × TSyntax `term) := []
      let mut aliases : List (Expr × Ident) := []
      let mut floatingPointers : List (String × Ident) := []
      for (name, value) in floating do
        let mut selectedAlias := none
        for (previous, id) in aliases do
          if ← isDefEq value previous then selectedAlias := some id
        let id ← match selectedAlias with
          | some id => pure id
          | none => do
            let fresh ← mkFreshId
            let id := mkIdent (Name.mkSimple ("veritile_storage_" ++ (fresh.toString.map (fun c => if c.isAlphanum then c else '_'))))
            aliases := aliases ++ [(value,id)]
            storageBindings := storageBindings ++ [(id,← exprToSyntax value)]
            pure id
        floatingPointers := floatingPointers ++ [(name,id)]
      let scalars ← bindings scalarTypeValue (← mkAppM ``Triton.CallSignature.scalars #[profile])
      let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM stringValue
      let readPorts ← (← listValues (← mkAppM ``Triton.CallSignature.readPorts #[profile])).mapM stringValue
      let writePorts ← (← listValues (← mkAppM ``Triton.CallSignature.writePorts #[profile])).mapM stringValue
      for ports in [readPorts,writePorts] do
        unless ports.eraseDups.length == ports.length && ports.all (fun name =>
            (pointers.map Prod.fst ++ floatingPointers.map Prod.fst).contains name) do
          throwError "memory ports must be distinct pointer parameter names"
      let names := pointers.map Prod.fst ++ floatingPointers.map Prod.fst ++ scalars.map Prod.fst ++ constants
      unless names.eraseDups.length == names.length do throwError "duplicate name in call signature"
      let sourceConstants ← bindings stringValue (← mkAppM ``Triton.CallSignature.sourceConstants #[profile])
      let imports ← bindings stringValue (← mkAppM ``Triton.CallSignature.unaryImports #[profile])
      let moduleNames := sourceConstants.map Prod.fst ++ imports.map Prod.fst
      unless moduleNames.eraseDups.length == moduleNames.length &&
          moduleNames.all (fun name => !names.contains name) do
        throwError "module bindings must be distinct from parameters and each other"
      liftMacroM <| SourceBindings.validateUnary imports
      let environment ← getEnv
      let sourceValues ← sourceConstants.mapM fun (name,source) =>
        liftMacroM <| SourceBindings.constant environment name source
      for stmt in stmts do rejectArgumentMutation (sourceValues.map Prod.fst) stmt.raw
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        let lowered ← SourceBindings.lowerUnary imports stmt.raw
        return (⟨← bindSourceValues sourceValues lowered⟩ : TSyntax `tritonStmt)
      for stmt in stmts do rejectBodyMetadata stmt.raw
      let mut values : List (Name × TSyntax `tritonExpr) := []
      let captured := if callValues.isSome then
        scalars.filterMap (fun (name,kind) =>
          if (kind matches .integer | .i32 | .i64) &&
              stmts.any (fun st => argumentSignedCast? st.raw == some (Name.mkSimple name))
          then some (Name.mkSimple name) else none)
        else []
      let mut fp32Values := Bool.false
      if let some args := callValues then
        for arg in args do
          let `(tritonCallBinding| $id:ident := $value:term) := arg
            | throwErrorAt arg "invalid call argument binding"
          let name := id.getId.eraseMacroScopes
          if values.any (fun p => p.1 == name) then throwErrorAt arg "duplicate call argument binding"
          let some (_, kind) := scalars.find? (fun p => p.1 == name.toString)
            | throwErrorAt arg "binding is not a runtime scalar parameter"
          let argument ← elabTerm value none
          let dtype ← inferType argument
          let expression ← match kind with
            | .fp32 => do
                unless ← isDefEq dtype (mkConst ``Real) do
                  throwErrorAt value "per-call FP32 literal must have a Real carrier"
                fp32Values := Bool.true
                liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.CallValue.fp32 $value:term : ℝ)))
            | .integer | .i32 | .i64 => do
                if ← isDefEq dtype (mkConst ``Nat) then
                  liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.CallValue.natural $value:term : Nat)))
                else if ← isDefEq dtype (mkConst ``Int) then
                  liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.CallValue.integer $value:term : Int)))
                else throwErrorAt value "per-call integer value must have a Nat or Int carrier"
            | .fp64 => throwErrorAt value "per-call FP64 specialization is unsupported; use runtime scalar binding"
          values := values ++ [(name, expression)]
        unless scalars.all (fun (name,kind) =>
            (kind matches .fp32 | .fp64) || values.any (fun p => p.1.toString == name)) do
          throwError "every integer runtime scalar needs an explicit call binding"
        for stmt in stmts do
          if (argumentSignedCast? stmt.raw).any captured.contains then continue
          rejectArgumentMutation (values.map Prod.fst) stmt.raw
      let stmts ← liftMacroM <| bindInvocationStatements values captured stmts
      let (constantBindings,prepared) ← liftMacroM <|
        RawBodyConstants.prepare (constants.map Name.mkSimple) [] stmts.toList
      let mut weakConstants : List (Name × TSyntax `tritonExpr) := []
      for constant in constants do
        let name := Name.mkSimple constant
        let id : TSyntax `term := ⟨mkIdent name⟩
        let value ← elabTerm id none
        let dtype ← inferType value
        if (← isDefEq dtype (mkConst ``Nat)) || (← isDefEq dtype (mkConst ``Bool)) ||
            (← isDefEq dtype (mkConst ``String)) then continue
        let expression ←
          if ← isDefEq dtype (mkConst ``Int) then
            liftMacroM `(tritonExpr| $(($id:term : Int)))
          else if ← isDefEq dtype (mkConst ``Real) then
            liftMacroM `(tritonExpr| $(($id:term : ℝ)))
          else if ← isDefEq dtype (mkConst ``Bool) then
            liftMacroM `(tritonExpr| $(($id:term : Bool)))
          else throwError "unsupported constexpr carrier; expected Nat, Int, Real, Bool or String"
        weakConstants := weakConstants ++ [(name,expression)]
      let stmts ← liftMacroM <| prepared.toArray.mapM fun st => do
        return (⟨← bindSourceValues weakConstants st.raw⟩ : TSyntax `tritonStmt)
      let parserEnv ← getEnv
      let stmts ← liftMacroM <| stmts.mapM fun stmt =>
        return (⟨← normalizeRawBody parserEnv (← PositionalBlockPointer.normalize (← RawBodyControl.normalize stmt.raw))⟩ : TSyntax `tritonStmt)
      for stmt in stmts do liftMacroM <| checkTupleReads stmt.raw
      let stmts ← liftMacroM <| stmts.mapM fun stmt =>
        return (⟨← PythonConstants.lower stmt.raw⟩ : TSyntax `tritonStmt)
      let (reads, writes) := Metadata.blockRegions stmts.toList
      for region in reads ++ writes do
        let `(term| $id:ident) := region
          | throwErrorAt region "call signature requires named external pointer parameters"
        unless (pointers.map Prod.fst ++ floatingPointers.map Prod.fst).contains id.getId.eraseMacroScopes.toString do
          throwErrorAt region "pointer parameter {id} is missing from the call signature"
      for (regions,ports) in [(reads,readPorts),(writes,writePorts)] do
        unless ports.isEmpty do
          for region in regions do
            let `(term| $id:ident) := region | throwErrorAt region "expected a pointer parameter"
            unless ports.contains id.getId.eraseMacroScopes.toString do
              throwErrorAt region "observed memory access is missing from the declared call ports"
      let term ← liftMacroM do
        let mut declarations : Array (TSyntax `tritonStmt) := #[← `(tritonStmt| tl.call_signature)]
        unless readPorts.isEmpty && writePorts.isEmpty do
          let explicitPorts (names : List String) : Array (TSyntax `term) :=
            (names.map (fun name => (⟨mkIdent (Name.mkSimple name)⟩ : TSyntax `term))).toArray
          let inputs := if readPorts.isEmpty then reads.toArray else explicitPorts readPorts
          let outputs := if writePorts.isEmpty then writes.toArray else explicitPorts writePorts
          declarations := declarations.push (← `(tritonStmt| tl.ports inputs=[$inputs,*], outputs=[$outputs,*]))
        let mut dtypeBindings : List (String × TSyntax `term) := []
        for (name, kind) in pointers do
          let id := mkIdent (Name.mkSimple name)
          let some dt ← dtypeSyntaxOfName? (SourceDType.dtypeName kind)
            | Macro.throwError "unsupported pointer dtype"
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = $dt:tritonDType))
          let dtypeTerm ← match kind with
            | .fp16 => pure (some (← `(FloatDType.fp16)))
            | .bf16 => pure (some (← `(FloatDType.bf16)))
            | .fp32 => pure (some (← `(FloatDType.fp32)))
            | .fp64 => pure (some (← `(FloatDType.real)))
            | .f8e4 => pure (some (← `(FloatDType.f8e4)))
            | .f8e5 => pure (some (← `(FloatDType.f8e5)))
            | _ => pure none
          if let some dtypeTerm := dtypeTerm then dtypeBindings := dtypeBindings ++ [(name,dtypeTerm)]
        for (name, dtype) in floatingPointers do
          let id := mkIdent (Name.mkSimple name)
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = storage_dtype($dtype:ident)))
          dtypeBindings := dtypeBindings ++ [(name,⟨dtype.raw⟩)]
        for (name, kind) in scalars do
          if values.any (fun p => p.1.toString == name) then continue
          if kind matches .i32 then
            Macro.throwError "int32 runtime arguments require an explicit call_args binding"
          let id := mkIdent (Name.mkSimple name)
          let dtName := match kind with
            | .fp32 => `tl.float32 | .fp64 => `tl.float64 | .integer | .i64 => `tl.int64
            | .i32 => `tl.int32
          let some dt ← dtypeSyntaxOfName? dtName
            | Macro.throwError "unsupported scalar dtype"
          declarations := declarations.push (← `(tritonStmt| tl.param $id:ident : $dt:tritonDType))
        RawBodyDTypeControl.emit dtypeBindings (fun selected => do
          let mut body := selected
          for (name, kind) in pointers do
            body ← RawBodyStores.lower (fun value => do
              if kind == .boolean then return ← SourceDType.storeValue kind value
              let some dtype ← dtypeSyntaxOfName? (SourceDType.dtypeName kind)
                | Macro.throwError "unsupported implicit store dtype"
              `(tritonExpr| tl.implicit_store_cast($value:tritonExpr, $dtype:tritonDType)))
              [Name.mkSimple name] body
          for (name, dtype) in floatingPointers do
            body ← RawBodyStores.lower (fun value =>
              `(tritonExpr| tl.implicit_store_cast($value:tritonExpr, storage_dtype($dtype:ident)))) [Name.mkSimple name] body
          let originalStores := selected.flatMap (fun st => storeValues st.raw)
          let convertedStores := body.flatMap (fun st => storeValues st.raw)
          unless originalStores.length == convertedStores.length do
            Macro.throwError "implicit store conversion changed the source store count"
          for (original, converted) in originalStores.zip convertedStores do
            if original == converted then
              Macro.throwErrorAt original "cannot determine tl.store destination dtype from the call signature"
          body ← RawBodyNestedLoads.lower body
          let emitted := declarations ++ body.toArray
          `(triton { $emitted* })) stmts.toList
      let term ← liftMacroM <| constantBindings.reverse.foldlM (init:=term) fun body (name,value) =>
        `(let $name:ident : Nat := $value:term; $body:term)
      let term ← liftMacroM <| storageBindings.reverse.foldlM (init:=term) fun body (name,value) =>
        `(let $name:ident : FloatDType := $value:term; $body:term)
      let term : TSyntax `term := ⟨← specializeRanges (constants.map Name.mkSimple) [] term.raw⟩
      let result ← elabTerm term expected
      let result ← if fp32Values then do
        synthesizeSyntheticMVarsNoPostponing
        let result ← instantiateMVars result
        checkFP32CallPrecision result
        pure result
        else pure result
      -- Provenance markers guide source typing only. Erase them after the
      -- typed operations (including integer-to-float conversion) are fixed.
      return result.replace fun e =>
        if e.isAppOfArity ``CallValue.fp32 1 ||
            e.isAppOfArity ``CallValue.natural 1 ||
            e.isAppOfArity ``CallValue.integer 1 then some e.appArg! else none

@[term_elab withSignature] def elaborateSignature : TermElab := fun stx expected =>
  elaborateCall ⟨stx[1]⟩ stx.getArgs.back! none expected

@[term_elab withCall] def elaborateInvocation : TermElab := fun stx expected => do
  let args := stx[4].getSepArgs.map (fun arg => (⟨arg⟩ : TSyntax `tritonCallBinding))
  elaborateCall ⟨stx[1]⟩ stx.getArgs.back! (some args) expected

end DSL.CallSignature


end VeriTile.Triton
