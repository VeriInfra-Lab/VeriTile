import VeriTile.Triton.DSL.SourceConstant
import VeriTile.Triton.DSL.NestedLoadAssignments
import VeriTile.Triton.DSL.ConstexprMask
import VeriTile.Triton.DSL.PythonConstants
import VeriTile.Triton.DSL.SourceIntrinsic
import VeriTile.Triton.DSL.OptionalString
import VeriTile.Triton.DSL.OptionalPointer
import VeriTile.Triton.DSL.TypedStores
import VeriTile.Triton.DSL.TypedStorageReduction
import VeriTile.Triton.DSL.PythonBody
import VeriTile.Triton.DSL.RankOneBlockPtr
import VeriTile.Triton.DSL.SourceFunctionStatements
import VeriTile.Triton.DSL.NarrowUnsigned
import VeriTile.Triton.DSL.SourceDType
import VeriTile.Triton.DSL.Constexpr
import Lean

/-! Call-site typing for a pasted Triton body. The signature describes pointer
storage and runtime scalar registers; it emits no kernel instruction. The body
uses ordinary Triton names, casts and implicit destination conversions. -/
namespace VeriTile.Triton

/-- Scalar carrier/precision supplied by the invocation. `integer` is the
mathematical integer channel, not a claim about an unannotated Python ABI width. -/
inductive ScalarType where
  | fp32 | fp64 | integer
  /-- Source ABI widths. Per-call integer specialization retains the source
  annotation; native range/refinement obligations remain with the caller IO. -/
  | i32 | i64 | u32 | u64

structure CallSignature where
  pointers : List (String × PointerType) := []
  /-- Pointer names whose full source dtype is supplied by the invocation.
  The source_dtype entrypoint checks every constructor, including integer and
  floating storage; it does not restrict the caller to one dtype. -/
  sourceDTypePointers : List String := []
  /-- Pointer parameters whose None/non-None specialization is supplied by the caller. -/
  optionalPointers : List String := []
  /-- Immutable Boolean constexpr inputs whose caller supplies a closed value.
  Specialize these before type analysis, so a Python argument may be a scalar
  in one call and a pointer in another without modifying either source branch. -/
  specializeBooleans : List String := []
  scalars : List (String × ScalarType) := []
  constexpr : List String := []
  /-- Keep the typed FP32 memory channel used by a caller's existing IO.
  This chooses an interpreter representation, not another source dtype or an
  extra cast. The default uses the profiled real carrier. -/
  typedFP32Storage : Bool := Bool.false
  /-- Select typed FP32 input/output channels individually when the existing
  IO mixes typed and profiled FP32 memory representations. No source cast is
  inserted and every named pointer still has source dtype FP32. -/
  typedFP32Pointers : List String := []
  /-- Keep a caller's already typed natural-valued integer memory channel.
  The pointer still has its declared source integer storage dtype. This is a
  mathematical carrier choice for the existing IO, not a native range proof;
  the caller must retain its storage-width/nonnegative refinement. -/
  naturalIntegerPointers : List String := []
  /-- Optional public memory roles when pointer rebinding prevents the legacy
  collector from recovering the original region. These are interface metadata,
  not instructions or claims that an access succeeds. -/
  readPorts : List String := []
  writePorts : List String := []
  /-- Checked original helper definitions supplied by the caller. Their
  signatures and raw bodies remain outside the pasted writer quotation. -/
  functions : List SourceFunction := []
  /-- Source import aliases for the checked CUDA finite-classification helper.
  Only the explicit finitef/isfinited targets accepted by FiniteHelperSource
  are supported. These bindings are metadata outside the pasted bodies. -/
  finiteImports : List (String × String) := []
  /-- Checked source import bindings for modeled unary intrinsics. -/
  unaryImports : List (String × String) := []
  /-- Exact binary intrinsic selected by the caller's source import binding. -/
  binaryImports : List (String × String) := []
  /-- Original immutable module-level `tl.constexpr` numeric literals.
  Names and literal text are checked against Python by source correspondence.
  They are source metadata, not function parameters or body instructions. -/
  sourceConstants : List (String × String) := []

namespace DSL.CallSignature
open Lean Elab Term Meta

syntax "with_triton_signature " term " in " "triton" tritonPythonBody : term
syntax "with_triton_signature " term " source_dtype " ident " := " term " in "
  "triton" tritonPythonBody : term
declare_syntax_cat tritonCallBinding
syntax ident " := " term : tritonCallBinding
syntax "with_triton_signature " term " pointer_presence " "[" tritonCallBinding,* "]" " in "
  "triton" tritonPythonBody : term
syntax "with_triton_call " term " call_args " "[" tritonCallBinding,* "]"
  " pointer_presence " "[" tritonCallBinding,* "]" " in " "triton" tritonPythonBody : term
syntax "with_triton_call " term " call_args " "[" tritonCallBinding,* "]" " in "
  "triton" tritonPythonBody : term
syntax "with_triton_call " term " call_args " "[" tritonCallBinding,* "]"
  " pointer_offsets " "[" tritonCallBinding,* "]" " in " "triton" tritonPythonBody : term

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

private def dtypeValue (value : Expr) : TermElabM Triton.PointerType := do
  let value ← whnf value
  let kinds : List (Name × Triton.PointerType) :=
    [(``Triton.PointerType.fp16,.fp16),(``Triton.PointerType.bf16,.bf16),
     (``Triton.PointerType.fp32,.fp32),(``Triton.PointerType.fp64,.fp64),
     (``Triton.PointerType.f8e4,.f8e4),(``Triton.PointerType.f8e5,.f8e5),
     (``Triton.PointerType.i8,.i8),(``Triton.PointerType.i16,.i16),
     (``Triton.PointerType.i32,.i32),(``Triton.PointerType.i64,.i64),
     (``Triton.PointerType.u8,.u8),(``Triton.PointerType.u16,.u16),
     (``Triton.PointerType.u32,.u32),(``Triton.PointerType.u64,.u64),
     (``Triton.PointerType.boolean,.boolean),
     (``Triton.PointerType.inputFloat,.inputFloat),(``Triton.PointerType.outputFloat,.outputFloat)]
  if let some (_,kind) := kinds.find? (fun p => value.isConstOf p.1) then return kind
  throwError "call signature dtypes must be statically known"

private def scalarTypeValue (value : Expr) : TermElabM ScalarType := do
  let value ← whnf value
  if value.isConstOf ``ScalarType.fp32 then return .fp32
  if value.isConstOf ``ScalarType.fp64 then return .fp64
  if value.isConstOf ``ScalarType.integer then return .integer
  if value.isConstOf ``ScalarType.i32 then return .i32
  if value.isConstOf ``ScalarType.i64 then return .i64
  if value.isConstOf ``ScalarType.u32 then return .u32
  if value.isConstOf ``ScalarType.u64 then return .u64
  throwError "unsupported runtime scalar type"

private def bindings {α : Type} (decode : Expr → TermElabM α) (value : Expr) :
    TermElabM (List (String × α)) := do
  (← listValues value).mapM fun pair => do
    let pair ← whnf pair
    unless pair.isAppOfArity ``Prod.mk 4 do throwError "invalid call signature binding"
    let args := pair.getAppArgs
    return (← stringValue args[2]!, ← decode args[3]!)

private partial def rejectBodyMetadata (raw : Syntax) : TermElabM Unit := do
  if let `(tritonStmt| tl.region $_:ident = $_:tritonDType $[, $_:ident = $_:tritonDType]*) := raw then
    throwErrorAt raw "pointer types belong in the call signature, outside the pasted body"
  if let `(tritonStmt| tl.param $_:ident : $_:tritonDType) := raw then
    throwErrorAt raw "runtime scalar types belong in the call signature, outside the pasted body"
  if let `(tritonStmt| $_:ident := $_:tritonExpr) := raw then
    throwErrorAt raw "use Python assignment syntax in the pasted body"
  if raw.getKind == ``tritonPythonWeakLiteral then
    throwErrorAt raw "source literal markers are internal compiler syntax"
  if let `(tritonExpr| $($_:term)) := raw then
    throwErrorAt raw "Lean antiquotes do not belong in the pasted body"
  for arg in raw.getArgs do rejectBodyMetadata arg

/-- Metadata-only view: every source method-cast spelling exposes its receiver
to the legacy region collector. The placeholder cast target is irrelevant to
region discovery. This syntax is never elaborated as the executable program. -/
private partial def sourcePortView (raw : Syntax) : MacroM Syntax := do
  let args ← raw.getArgs.mapM sourcePortView
  let rebuilt := match raw with
    | .node info kind _ => Syntax.node info kind args
    | _ => raw
  let e : TSyntax `tritonExpr := ⟨rebuilt⟩
  -- The parser has dedicated load+cast productions whose load is not a child
  -- expression. Expose that memory operation explicitly to the collector.
  match e with
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident.$_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident.$_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | _ => pure ()
  let k := raw.getKind
  if [``tritonMethodCast, ``tritonMethodCastDTypeIdent,
      ``tritonMethodCastElementTy, ``tritonMethodCastElementTyIdent,
      ``tritonMethodRoundCast].contains k && !args.isEmpty then
    let receiver : TSyntax `tritonExpr := ⟨args[0]!⟩
    return (← `(tritonExpr| tl.cast($receiver, tl.int64))).raw
  return rebuilt

private partial def bindSourceValues (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) :
    MacroM Syntax := do
  -- These terms already belong to the caller's Lean context. Replacing a
  -- name inside one would inject a Triton expression into a Lean term.
  if let `(tritonExpr| $($_:term)) := raw then return raw
  -- A static-loop bound is already a caller-owned Lean term. Its children
  -- must not be rebound as source identifiers after constexpr normalization.
  if let `(tritonStmt| for $i:ident in tl.static_range($($n:term)) { $body:tritonStmt* }) := raw then
    let body ← body.mapM fun st => do
      return (⟨← bindSourceValues values st.raw⟩ : TSyntax `tritonStmt)
    return (← `(tritonStmt| for $i in tl.static_range($($n)) { $body* })).raw
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

private partial def identifierNames (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifierNames

private partial def hasComma (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal.trimAscii.toString == ",") || raw.getArgs.any hasComma

/-- Include tuple-call and positional-reduction assignments as well as the
ordinary assignment production. Keyword arguments are not assignment targets. -/
private def assignmentTargets (raw : Syntax) : List Name := Id.run do
  let args := raw.getArgs
  unless args[0]?.any Syntax.isIdent do return []
  let some position := args.findIdx? (fun arg => arg.isAtom &&
      ["=",":=","+=","-=","*=","/="].contains arg.getAtomVal.trimAscii.toString) | return []
  let left := args.extract 0 position
  unless (raw.getKind.toString.splitOn "tritonStmt").length > 1 || left.any hasComma do return []
  return left.toList.flatMap identifierNames

private partial def rejectArgumentMutation (names : List Name) (raw : Syntax)
    (message : String := "per-call specialization does not yet support reassigned scalar arguments") : TermElabM Unit := do
  let reject (id : Ident) : TermElabM Unit := do
    if names.contains id.getId.eraseMacroScopes then
      throwErrorAt raw "{message}"
  for target in assignmentTargets raw do reject (mkIdent target)
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
  for arg in raw.getArgs do rejectArgumentMutation names arg message

/-- Specialize call inputs in statement order. A direct integer or offset-pointer
assignment reads its entry value on the RHS and binds the original source LHS.
All later uses read that local register, including after a zero-trip loop.
A conditional first assignment specializes each arm with its continuation.
Loop, tuple, augmented and FP parameter writes remain rejected. -/
private partial def bindCallStatements (rebindable : List Name)
    (values : List (Name × TSyntax `tritonExpr))
    (stmts : Array (TSyntax `tritonStmt)) : TermElabM (Array (TSyntax `tritonStmt)) := do
  let mut current := values
  let mut result := #[]
  for i in [:stmts.size] do
    let st := stmts[i]!
    if let `(tritonStmt| $id:ident = $rhs:tritonExpr) := st then
      let name := id.getId.eraseMacroScopes
      if rebindable.contains name && current.any (fun value => value.1 == name) then
        let rhs : TSyntax `tritonExpr := ⟨← liftMacroM <| bindSourceValues current rhs.raw⟩
        result := result.push (← liftMacroM `(tritonStmt| $id:ident = $rhs:tritonExpr))
        current := current.filter (fun value => value.1 != name)
        continue
    let branches : Option (TSyntax `tritonExpr × Array (TSyntax `tritonStmt) × Array (TSyntax `tritonStmt)) :=
      match st with
      | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => some (cond,yes,no)
      | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* }) => some (cond,yes,#[])
      | _ => none
    if let some (cond,yes,no) := branches then
      -- Only take this path when a still-specialized caller value is assigned.
      -- Each recursive branch retains its own binding environment, including
      -- the unchanged entry value on a path which does not assign it.
      let rec writes (raw : Syntax) : Bool :=
        (assignmentTargets raw).any (fun n => current.any (fun p => p.1 == n)) || raw.getArgs.any writes
      if yes.any (fun st => writes st.raw) || no.any (fun st => writes st.raw) then
        let cond : TSyntax `tritonExpr := ⟨← liftMacroM <| bindSourceValues current cond.raw⟩
        let rest := stmts.extract (i+1) stmts.size
        let yes ← bindCallStatements rebindable current (yes ++ rest)
        let no ← bindCallStatements rebindable current (no ++ rest)
        return result.push (← liftMacroM `(tritonStmt| if $cond { $yes* } else { $no* }))
    rejectArgumentMutation (current.map Prod.fst) st.raw
    result := result.push (⟨← liftMacroM <| bindSourceValues current st.raw⟩ : TSyntax `tritonStmt)
  return result

private partial def assignedNames (raw : Syntax) : List Name := Id.run do
  let mut names := assignmentTargets raw
  if raw.getNumArgs > 1 && raw[0].isAtom && raw[1].isIdent &&
      ["for", "tl.for"].contains raw[0].getAtomVal.trimAscii.toString then
    names := raw[1].getId.eraseMacroScopes :: names
  return names ++ raw.getArgs.toList.flatMap assignedNames

private partial def rejectConstexprIteratorShadow (constants : List String)
    (raw : Syntax) : TermElabM Unit := do
  if raw.getNumArgs > 1 && raw[0].isAtom && raw[1].isIdent &&
      ["for", "tl.for"].contains raw[0].getAtomVal.trimAscii.toString &&
      constants.contains raw[1].getId.eraseMacroScopes.toString then
    throwErrorAt raw "loop iterator shadows a constexpr parameter"
  for arg in raw.getArgs do rejectConstexprIteratorShadow constants arg

private partial def staticNatBound (constants assigned : List Name)
    (e : TSyntax `tritonExpr) : TermElabM (Option (TSyntax `term)) := do
  match e with
  | `(tritonExpr| ($inner:tritonExpr)) => staticNatBound constants assigned inner
  | `(tritonExpr| $n:num) => return some (← `(($n : Nat)))
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) =>
      -- Supplied immutable Nat dimensions can determine a range endpoint.
      -- Mutable source locals remain on the dynamic path. Preserve the
      -- explicit zero-divisor path used by literal source expressions.
      if let `(tritonExpr| $zero:num) := b then
        if zero.getNat == 0 then return none
      let some a ← staticNatBound constants assigned a | return none
      let some b ← staticNatBound constants assigned b | return none
      return some (← `((((($a : Nat) + ($b : Nat) - 1) / ($b : Nat)) : Nat)))
  | `(tritonExpr| $($term:term)) =>
      let value ← elabTerm term none
      if ← isDefEq (← inferType value) (mkConst ``Nat) then return some term else return none
  | `(tritonExpr| $id:ident) =>
      let name := id.getId.eraseMacroScopes
      unless constants.contains name && !assigned.contains name do return none
      let term : TSyntax `term := ⟨id.raw⟩
      let value ← elabTerm term none
      if ← isDefEq (← inferType value) (mkConst ``Nat) then return some term else return none
  | _ => return none

/-- Retain a checked arange bound through immutable caller aliases. Inspect the
actual constant before reduction; an arbitrary symbolic Nat or a same-named
local function cannot manufacture the checked integer profile. -/
private partial def checkedArangeTerm? (value : Expr) : TermElabM (Option (TSyntax `term)) := do
  let value ← instantiateMVars value
  if let .fvar id := value then
    if let some rhs := (← id.getDecl).value? then return ← checkedArangeTerm? rhs
  if let .letE _ _ rhs body _ := value then
    return ← checkedArangeTerm? (body.instantiate1 rhs)
  if let .mdata _ inner := value then return ← checkedArangeTerm? inner
  if value.isAppOfArity ``VeriTile.Triton.Random.i32ArangeSize 2 then
    let args := value.getAppArgs
    let size ← exprToSyntax args[0]!
    let proof ← exprToSyntax args[1]!
    return some (← `(_root_.VeriTile.Triton.Random.i32ArangeSize $size $proof))
  return none

/-- A scalar signed full of a checked natural constant is a known forward
range stride. Keep its source assignment, and specialize only the loop bound.
Only a unique preceding top-level definition is accepted; any later, nested,
tuple or iterator write invalidates the fact for the entire body. -/
private partial def substituteRangeStrides (known : List (Name × TSyntax `tritonExpr))
    (raw : Syntax) : MacroM Syntax := do
  let args ← raw.getArgs.mapM (substituteRangeStrides known)
  let rebuilt := match raw with
    | .node info kind _ => Syntax.node info kind args
    | _ => raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $step:ident) { $body:tritonStmt* }) := rebuilt then
    if let some (_, value) := known.find? (fun p => p.1 == step.getId.eraseMacroScopes) then
      return (← `(tritonStmt| for $i in range($a, $b, $value:tritonExpr) { $body* })).raw
  return rebuilt

private def specializeImmutableRangeStrides (body : Array (TSyntax `tritonStmt)) :
    TermElabM (Array (TSyntax `tritonStmt)) := do
  let written := body.toList.flatMap (fun s => assignedNames s.raw)
  let mut known : List (Name × TSyntax `tritonExpr) := []
  let mut out := #[]
  for stmt in body do
    out := out.push (⟨← liftMacroM <| substituteRangeStrides known stmt.raw⟩ : TSyntax `tritonStmt)
    if let `(tritonStmt| $id:ident = tl.full([], $value:tritonExpr, tl.int64)) := stmt then
      let name := id.getId.eraseMacroScopes
      unless (written.filter (· == name)).length == 1 do continue
      let `(tritonExpr| $($term:term)) := value | continue
      let term ← elabTerm term (some (mkConst ``Nat))
      let checked ← checkedArangeTerm? term
      let closed ← getNatValue? (← whnf term)
      unless checked.isSome || closed.any (· < 2^31) do continue
      known := (name,value) :: known
  return out


/-- Finite literal results of an immutable string caller expression. The union
is conservative: correlations between caller conditions are never assumed. -/
private partial def finiteStringValues (value : Expr) : TermElabM (List String) := do
  let value ← instantiateMVars value
  if let .lit (.strVal s) := value then return [s]
  if let .fvar id := value then
    if let some rhs := (← id.getDecl).value? then return ← finiteStringValues rhs
  if let .letE _ _ rhs body _ := value then
    return ← finiteStringValues (body.instantiate1 rhs)
  if let .mdata _ inner := value then return ← finiteStringValues inner
  if value.isAppOfArity ``ite 5 then
    let args := value.getAppArgs
    return ((← finiteStringValues args[3]!) ++ (← finiteStringValues args[4]!)).eraseDups
  let reduced ← whnf value
  if let .lit (.strVal s) := reduced then return [s]
  throwError "string constexpr requires a literal or a finite conditional of string literals"

private abbrev StringChoices := List (Name × List String)

/-- Eliminate only string tests forced by the actual caller's finite literal
choices and preceding tests on that same immutable name. This lets dtype flow
see exhaustive if/elif source paths without inventing an else for other values. -/
private partial def specializeStringBlock (choices : StringChoices)
    (stmts : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  let mut result := []
  for st in stmts do
    let branches : Option (TSyntax `tritonExpr × Array (TSyntax `tritonStmt) ×
        Array (TSyntax `tritonStmt) × Bool) := match st with
      | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          some (cond,yes,no,Bool.true)
      | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* }) =>
          some (cond,yes,#[],Bool.false)
      | _ => none
    if let some (cond,yes,no,hasElse) := branches then
      let test : Option (Name × String × Bool) := match cond with
        | `(tritonExpr| $id:ident == $s:str) => some (id.getId.eraseMacroScopes,s.getString,Bool.true)
        | `(tritonExpr| $id:ident != $s:str) => some (id.getId.eraseMacroScopes,s.getString,Bool.false)
        | _ => none
      if let some (name,literal,equal) := test then
        if let some (_,values) := choices.find? (fun p => p.1 == name) then
          let yesValues := values.filter (fun s => (s == literal) == equal)
          let noValues := values.filter (fun s => (s == literal) != equal)
          let replace (values : List String) := choices.map (fun p =>
            if p.1 == name then (name,values) else p)
          if yesValues.isEmpty then
            result := result ++ (← specializeStringBlock (replace noValues) no.toList)
            continue
          if noValues.isEmpty then
            result := result ++ (← specializeStringBlock (replace yesValues) yes.toList)
            continue
          let yes := (← specializeStringBlock (replace yesValues) yes.toList).toArray
          let no := (← specializeStringBlock (replace noValues) no.toList).toArray
          result := result ++ [← if hasElse then `(tritonStmt| if $cond { $yes* } else { $no* })
            else `(tritonStmt| if $cond { $yes* })]
          continue
      let yes := (← specializeStringBlock choices yes.toList).toArray
      let no := (← specializeStringBlock choices no.toList).toArray
      result := result ++ [← if hasElse then `(tritonStmt| if $cond { $yes* } else { $no* })
        else `(tritonStmt| if $cond { $yes* })]
      continue
    match st with
    | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← specializeStringBlock choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i in range($a,$b,$c) { $body* })]
    | `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← specializeStringBlock choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i in range($a,$b) { $body* })]
    | `(tritonStmt| for $i:ident in range($a:tritonExpr) { $body:tritonStmt* }) =>
        let body := (← specializeStringBlock choices body.toList).toArray
        result := result ++ [← `(tritonStmt| for $i in range($a) { $body* })]
    | _ => result := result ++ [st]
  return result

/-- Python integer constexpr arithmetic is signed before an explicit int64
cast. In particular, `BT - 1` at BT=0 is -1, not truncated Nat subtraction.
This bounded evaluator accepts only immutable constexpr integers and +,-,*;
runtime values keep their original source-width arithmetic. -/
private partial def staticPythonInteger (constants assigned : List Name)
    (expression : TSyntax `tritonExpr) : TermElabM (Option (TSyntax `term)) := do
  match expression with
  | `(tritonExpr| $n:num) => return some (← `(($n : Int)))
  | `(tritonExpr| ($inner:tritonExpr)) => staticPythonInteger constants assigned inner
  | `(tritonExpr| -$inner:tritonExpr) =>
      let some value ← staticPythonInteger constants assigned inner | return none
      return some (← `(-($value : Int)))
  | `(tritonExpr| $id:ident) =>
      let name := id.getId.eraseMacroScopes
      unless constants.contains name && !assigned.contains name do return none
      let term : TSyntax `term := ⟨id.raw⟩
      let value ← elabTerm term none
      let carrier ← inferType value
      if ← isDefEq carrier (mkConst ``Nat) then return some (← `(Int.ofNat ($term : Nat)))
      if ← isDefEq carrier (mkConst ``Int) then return some term
      return none
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let some left ← staticPythonInteger constants assigned a | return none
      let some right ← staticPythonInteger constants assigned b | return none
      match expression with
      | `(tritonExpr| $_:tritonExpr + $_:tritonExpr) => return some (← `(($left : Int) + ($right : Int)))
      | `(tritonExpr| $_:tritonExpr - $_:tritonExpr) => return some (← `(($left : Int) - ($right : Int)))
      | _ => return some (← `(($left : Int) * ($right : Int)))
  | _ => return none

private partial def hasIntegerArithmetic (expression : TSyntax `tritonExpr) : Bool :=
  match expression with
  | `(tritonExpr| ($inner:tritonExpr)) => hasIntegerArithmetic inner
  | `(tritonExpr| $_:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr - $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr * $_:tritonExpr) => Bool.true
  | _ => Bool.false

/-- Evaluate immutable Nat constexpr quotients at specialization time, as the
source JIT does for valid compile-time dimensions. Literal zero divisors keep
an explicit divide node; runtime and mutable names keep the runtime path.
The caller's native shape contract still requires valid nonzero dimensions. -/
private partial def foldStaticNatQuotients (constants assigned : List Name)
    (raw : Syntax) : TermElabM Syntax := do
  let args ← raw.getArgs.mapM (foldStaticNatQuotients constants assigned)
  let next := match raw with | .node info kind _ => Syntax.node info kind args | _ => raw
  let expression : TSyntax `tritonExpr := ⟨next⟩
  if let `(tritonExpr| tl.cast($value:tritonExpr, tl.int64)) := expression then
    if hasIntegerArithmetic value then
      if let some integer ← staticPythonInteger constants assigned value then
        return (← liftMacroM `(tritonExpr| tl.cast($(($integer : Int)), tl.int64))).raw
  if let `(tritonExpr| $dividend:tritonExpr // $divisor:tritonExpr) := expression then
    if let `(tritonExpr| $zero:num) := divisor then
      if zero.getNat == 0 then return next
    if let some lhs ← staticNatBound constants assigned dividend then
      if let some rhs ← staticNatBound constants assigned divisor then
        return (← liftMacroM `(tritonExpr| $((($lhs:term : Nat) / ($rhs:term : Nat) : Nat)))).raw
  if let `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) := expression then
    if let `(tritonExpr| $zero:num) := b then
      if zero.getNat == 0 then return next
    if let some lhs ← staticNatBound constants assigned a then
      if let some rhs ← staticNatBound constants assigned b then
        return (← liftMacroM `(tritonExpr| $((((($lhs : Nat) + ($rhs : Nat) - 1) / ($rhs : Nat)) : Nat)))).raw
  if let `(tritonStmt| for $i:ident in tl.static_range($bound:tritonExpr) { $body:tritonStmt* }) := next then
    if let some n ← staticNatBound constants assigned bound then
      return (← liftMacroM `(tritonStmt| for $i in tl.static_range($($n)) { $body* })).raw
    throwErrorAt bound "tl.static_range requires an immutable Nat constexpr bound"
  return next

/-- Normalize only signed range endpoints whose source explicitly widens an
immutable Nat constexpr. Casts in values stay visible to source ABI/RNG analysis. -/
private def staticSignedRangeBound (constants assigned : List Name)
    (e : TSyntax `tritonExpr) : TermElabM (TSyntax `tritonExpr) := do
  if let `(tritonExpr| tl.cast($value:tritonExpr, tl.int64)) := e then
    if let some value ← staticNatBound constants assigned value then
      return ← liftMacroM `(tritonExpr| $((Int.ofNat ($value : Nat) : Int)))
  return e

/-- A Nat caller/constexpr value has already been substituted by this pass.
Literal-only source ranges retain their existing dynamic AST representation. -/
private partial def suppliedRangeBound (raw : Syntax) : Bool :=
  if let `(tritonExpr| $($_:term)) := raw then Bool.true
  else raw.getArgs.any suppliedRangeBound

/-- A per-call size has already been supplied before executing the body.
Select the existing constant-bound range constructor only for known immutable
Nat bounds. Runtime locals and signed casts retain their dynamic range path. -/
private partial def specializeRanges (constants assigned : List Name) (raw : Syntax) : TermElabM Syntax := do
  let args ← raw.getArgs.mapM (specializeRanges constants assigned)
  let next := match raw with | .node info kind _ => Syntax.node info kind args | _ => raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr) { $body:tritonStmt* }) := next then
    let aa ← staticSignedRangeBound constants assigned a
    let bb ← staticSignedRangeBound constants assigned b
    if aa.raw != a.raw || bb.raw != b.raw then
      return (← liftMacroM `(tritonStmt| for $i in range($aa, $bb) { $body* })).raw
    -- Preserve the existing dynamic constructor for literal-only source loops.
    -- A specialized caller endpoint (e.g. range(0,T)) is an explicit antiquote
    -- by this pass; source literals alone need no representation change.
    if suppliedRangeBound a.raw || suppliedRangeBound b.raw then
      if let some start ← staticNatBound constants assigned a then
        if let some stop ← staticNatBound constants assigned b then
          return (← liftMacroM `(tritonStmt| for $i in range($($start), $($stop), $((1 : Nat))) { $body* })).raw
  if let `(tritonStmt| for $i:ident in range($n:tritonExpr) { $body:tritonStmt* }) := next then
    if let some n ← staticNatBound constants assigned n then
      return (← liftMacroM `(tritonStmt| tl.for $i in $($n) { $body* })).raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) := next then
    let aa ← staticSignedRangeBound constants assigned a
    let bb ← staticSignedRangeBound constants assigned b
    if aa.raw != a.raw || bb.raw != b.raw then
      return (← liftMacroM `(tritonStmt| for $i in range($aa, $bb, $c) { $body* })).raw
    if let some a ← staticNatBound constants assigned a then
      if let some b ← staticNatBound constants assigned b then
        if let some c ← staticNatBound constants assigned c then
          return (← liftMacroM `(tritonStmt| for $i in range($($a), $($b), $($c)) { $body* })).raw
  return next

/-- Literal specialization cannot erase the source argument's FP32 rounding.
Accept it only when elaboration already placed every use under FP32 evaluation.
Other contexts use the ordinary runtime register interface. -/
private partial def checkFP32CallPrecision (expression : Expr) (fp32 : Bool := Bool.false) : TermElabM Unit := do
  if expression.isAppOf ``TypedStores.fp32CallValue then
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

/-- Rank-one pointer lowering consumes its checked singleton axes first.
Remaining singleton tuples belong to ordinary block pointers; preserve their
axis as a static Nat list for the shared load/store compiler. -/
private partial def normalizeBoundaryTuples (raw : Syntax) : MacroM Syntax := do
  if let `(tritonMemKwarg| boundary_check=($axisIndex:tritonExpr,)) := raw then
    let axisTerm ← natDimTerm "boundary_check" axisIndex
    let axes ← `(term| ([$axisTerm] : List Nat))
    return (← `(tritonMemKwarg| boundary_check=$axes:term)).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM normalizeBoundaryTuples)
  | _ => return raw

private partial def referencesSourceImports (names : List String) (raw : Syntax) : Bool :=
  (raw.isIdent && names.contains raw.getId.eraseMacroScopes.toString) ||
    raw.getArgs.any (referencesSourceImports names)

set_option maxRecDepth 4000 in
/-- A verification call specializes immutable runtime size values supplied by
the caller. Their source ABI remains in `scalars`, never in `constexpr`.
Nat/Int carriers use the existing mathematical index projection;
native signed-width bounds must still be established by the task's IO/contract. -/
private def elaborateCallCore (profile : TSyntax `term) (stmts : Array (TSyntax `tritonStmt))
    (callValues : Option (Array (TSyntax `tritonCallBinding)))
    (pointerOffsets : Option (Array (TSyntax `tritonCallBinding)) := none)
    (pointerPresence : Option (Array (TSyntax `tritonCallBinding)) := none) : TermElabM Expr := do
      let profile ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
      let dtypePointers ← listValues (← mkAppM ``Triton.CallSignature.sourceDTypePointers #[profile])
      unless dtypePointers.isEmpty do
        throwError "source dtype pointer requires its source_dtype invocation binding"
      let pointers ← bindings dtypeValue (← mkAppM ``Triton.CallSignature.pointers #[profile])
      let optionalNames ← (← listValues
        (← mkAppM ``Triton.CallSignature.optionalPointers #[profile])).mapM stringValue
      unless optionalNames.eraseDups.length == optionalNames.length &&
          optionalNames.all (fun name => pointers.any (fun p => p.1 == name)) do
        throwError "optional pointers must be distinct declared pointer names"
      let mut presence : OptionalPointer.Presence := []
      for arg in pointerPresence.getD #[] do
        let `(tritonCallBinding| $id:ident := $value:term) := arg
          | throwErrorAt arg "invalid pointer presence binding"
        let name := id.getId.eraseMacroScopes
        unless optionalNames.contains name.toString do
          throwErrorAt id "presence binding requires a declared optional pointer"
        if presence.any (fun entry => entry.1 == name) then
          throwErrorAt id "duplicate pointer presence binding"
        let _ ← elabTermEnsuringType value (mkConst ``Bool)
        let expression ← liftMacroM `(tritonExpr| $(($value:term : Bool)))
        presence := presence ++ [(name,expression)]
      unless optionalNames.all (fun name => presence.any (fun entry => entry.1.toString == name)) do
        throwError "every optional pointer needs an explicit caller presence binding"

      let storage ← whnf (← mkAppM ``Triton.CallSignature.typedFP32Storage #[profile])
      let typedFP32 := storage.isConstOf ``Bool.true
      unless typedFP32 || storage.isConstOf ``Bool.false do
        throwError "typed FP32 storage selection must be statically known"
      let typedFP32Names ← (← listValues (← mkAppM ``Triton.CallSignature.typedFP32Pointers #[profile])).mapM stringValue
      unless typedFP32Names.eraseDups.length == typedFP32Names.length &&
          typedFP32Names.all (fun name => pointers.any (fun p => p.1 == name && p.2 == .fp32)) do
        throwError "typed FP32 pointers must be distinct FP32 pointer parameter names"
      let usesTypedFP32 := typedFP32 || !typedFP32Names.isEmpty
      -- Family aliases are caller-owned FloatDType values, not constexpr
      -- scalar expressions and not a promise that every case is FP32.
      for (kind, familyName) in [(PointerType.inputFloat, "dtype"), (PointerType.outputFloat, "out_dtype")] do
        if pointers.any (fun p => p.2 == kind) then
          if familyName == "dtype" && usesTypedFP32 then
            throwError "inputFloat metadata cannot share the fixed typed-FP32 dtype alias"
          let value ← elabTerm (mkIdent (Name.mkSimple familyName)) none
          unless (← whnf (← inferType value)).isConstOf ``Triton.FloatDType do
            throwError "floating pointer family requires an already typed FloatDType caller alias"
      let naturalIntegerNames ← (← listValues
        (← mkAppM ``Triton.CallSignature.naturalIntegerPointers #[profile])).mapM stringValue
      let integerStorage : List PointerType := [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64]
      unless naturalIntegerNames.eraseDups.length == naturalIntegerNames.length &&
          naturalIntegerNames.all (fun name => pointers.any
            (fun p => p.1 == name && integerStorage.contains p.2)) do
        throwError "natural integer pointers must be distinct integer pointer parameter names"
      for name in naturalIntegerNames do
        let id := mkIdent (Name.mkSimple name)
        let value ← elabTerm (⟨id.raw⟩ : TSyntax `term) none
        let type ← whnf (← inferType value)
        unless type.isAppOfArity ``Triton.Region 1 &&
            type.getAppArgs[0]!.isConstOf ``Triton.TileDType.nat do
          throwError "natural integer pointer requires an already typed Region .nat caller"
      let scalars ← bindings scalarTypeValue (← mkAppM ``Triton.CallSignature.scalars #[profile])
      let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM stringValue
      let readPorts ← (← listValues (← mkAppM ``Triton.CallSignature.readPorts #[profile])).mapM stringValue
      let writePorts ← (← listValues (← mkAppM ``Triton.CallSignature.writePorts #[profile])).mapM stringValue
      for ports in [readPorts,writePorts] do
        unless ports.eraseDups.length == ports.length && ports.all (fun name => pointers.any (fun p => p.1 == name)) do
          throwError "memory ports must be distinct pointer parameter names"
      -- Dtype dataflow must see the same structured if/elif tree as execution.
      let stmts ← liftMacroM <| stmts.mapM fun st => do
        return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
      let names := pointers.map Prod.fst ++ scalars.map Prod.fst ++ constants
      unless names.eraseDups.length == names.length do throwError "duplicate name in call signature"
      if usesTypedFP32 && names.contains "dtype" then
        throwError "typed FP32 storage reserves the metadata dtype alias"
      for stmt in stmts do rejectBodyMetadata stmt.raw
      let sourceFunctions ← listValues (← mkAppM ``Triton.CallSignature.functions #[profile])
      let helperNames ← sourceFunctions.mapM fun function => do
        stringValue (← mkAppM ``Triton.SourceFunction.name #[function])
      unless helperNames.eraseDups.length == helperNames.length &&
          helperNames.all (fun helper => !names.contains helper) do
        throwError "source helper names must be distinct from call parameters"
      let finiteImports ← bindings stringValue (← mkAppM ``Triton.CallSignature.finiteImports #[profile])
      let unaryImports ← bindings stringValue (← mkAppM ``Triton.CallSignature.unaryImports #[profile])
      let binaryImports ← bindings stringValue (← mkAppM ``Triton.CallSignature.binaryImports #[profile])
      liftMacroM <| SourceIntrinsic.validateBinary binaryImports
      liftMacroM <| SourceIntrinsic.validate unaryImports
      let importedNames := finiteImports.map Prod.fst ++ unaryImports.map Prod.fst ++ binaryImports.map Prod.fst
      unless importedNames.eraseDups.length == importedNames.length &&
          importedNames.all (fun name => !names.contains name && !helperNames.contains name) do
        throwError "source import aliases must be distinct from parameters and helpers"
      let environment ← getEnv
      let sourceConstants ← bindings stringValue (← mkAppM ``Triton.CallSignature.sourceConstants #[profile])
      let globalNames := sourceConstants.map Prod.fst
      unless globalNames.eraseDups.length == globalNames.length &&
          globalNames.all (fun name => !names.contains name && !helperNames.contains name && !importedNames.contains name) do
        throwError "source constants must be distinct from parameters, helpers and imports"
      let checkedConstants ← sourceConstants.mapM fun (name, source) =>
        liftMacroM <| SourceConstant.check environment name source
      for stmt in stmts do
        rejectArgumentMutation (globalNames.map Name.mkSimple) stmt.raw "source constants cannot be rebound"
      let checkedFunctions ← sourceFunctions.mapM fun function => do
        let functionName ← stringValue (← mkAppM ``Triton.SourceFunction.name #[function])
        let parameters ← (← listValues
          (← mkAppM ``Triton.SourceFunction.parameters #[function])).mapM stringValue
        let source ← stringValue (← mkAppM ``Triton.SourceFunction.body #[function])
        let constants ← (← listValues (← mkAppM ``Triton.SourceFunction.constexpr #[function])).mapM stringValue
        liftMacroM <| SourceFunction.check environment functionName parameters source (helperNames ++ importedNames) constants
      for stmt in stmts do
        rejectArgumentMutation ((helperNames ++ importedNames).map Name.mkSimple) stmt.raw
          (if importedNames.isEmpty then "source helper names cannot be rebound"
           else "source helper and import names cannot be rebound")
      -- Helpers which reference these explicit extern imports are validated
      -- by the existing exact dtype-dispatch checker. Ordinary pure helpers
      -- keep their existing capture-free expression expansion.
      let usesFiniteImport := fun (fn : SourceFunction.Checked) =>
        fn.results.any (fun value => referencesSourceImports (finiteImports.map Prod.fst) value.raw) ||
        fn.assignments.any (fun statement => referencesSourceImports (finiteImports.map Prod.fst) statement.raw)
      for fn in checkedFunctions do
        if usesFiniteImport fn && (!fn.assignments.isEmpty || fn.results.size != 1 || !fn.constexpr.isEmpty) then
          throwError "finite extern helper requires a single non-constexpr return expression"
      let finiteFunctions := checkedFunctions.filter usesFiniteImport
      let ordinaryFunctions := checkedFunctions.filter (fun fn => !usesFiniteImport fn)
      let finiteDeclarations ← liftMacroM do
        let mut declarations : Array (TSyntax `tritonStmt) := #[]
        for (name,target) in finiteImports do
          let aliasName := mkIdent (Name.mkSimple name)
          let imported := mkIdent (target.toName)
          declarations := declarations.push (← `(tritonStmt| tl.source_alias $aliasName:ident = $imported:ident))
        for fn in finiteFunctions do
          let [parameter] := fn.parameters
            | Macro.throwError "finite source helper requires one parameter"
          let helper := mkIdent fn.name
          let parameter := mkIdent parameter
          let expression := fn.result
          declarations := declarations.push (← `(tritonStmt| tl.source_helper $helper:ident($parameter:ident) { return $expression:tritonExpr }))
        -- Validate imports and helper expressions even if a helper is unused.
        let _ ← FiniteHelperSource.prepare declarations.toList names
        pure declarations
      let stmts ← liftMacroM <| SourceFunction.expandStatements ordinaryFunctions stmts names
      let stmts ← liftMacroM <| NarrowUnsigned.lower stmts
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        return (⟨← SourceIntrinsic.lower unaryImports stmt.raw⟩ : TSyntax `tritonStmt)
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        return (⟨← SourceIntrinsic.lowerBinary binaryImports stmt.raw⟩ : TSyntax `tritonStmt)
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        return (⟨← bindSourceValues checkedConstants stmt.raw⟩ : TSyntax `tritonStmt)
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        return (⟨← PythonConstants.lower stmt.raw⟩ : TSyntax `tritonStmt)
      let mut values : List (Name × TSyntax `tritonExpr) := []
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
                liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.TypedStores.fp32CallValue $value:term : ℝ)))
            | .i32 | .i64 | .u32 | .u64 | .integer => do
                let sourceKind ← match kind with
                  | .i32 => `(VeriTile.Triton.Random.IntegerType.i32)
                  | .i64 => `(VeriTile.Triton.Random.IntegerType.i64)
                  | .u32 => `(VeriTile.Triton.Random.IntegerType.u32)
                  | _ => `(VeriTile.Triton.Random.IntegerType.u64)
                if ← isDefEq dtype (mkConst ``Nat) then
                  if kind matches .integer then
                    liftMacroM `(tritonExpr| $(($value:term : Nat)))
                  else
                    liftMacroM `(tritonExpr| $((_root_.VeriTile.Triton.Random.sourceNatCallValue $sourceKind:term $value:term : Nat)))
                else if ← isDefEq dtype (mkConst ``Int) then
                  if kind matches .u32 | .u64 then
                    throwErrorAt value "unsigned per-call integer value must have a Nat carrier"
                  if kind matches .integer then
                    liftMacroM `(tritonExpr| $(($value:term : Int)))
                  else
                    liftMacroM `(tritonExpr| $((_root_.VeriTile.Triton.Random.sourceIntCallValue $sourceKind:term $value:term : Int)))
                else throwErrorAt value "per-call integer value must have a Nat or Int carrier"
            | .fp64 => throwErrorAt value "per-call FP64 literal specialization is unsupported; use runtime scalar binding"
          values := values ++ [(name, expression)]
        unless scalars.all (fun (name,kind) =>
            (kind matches .fp32 | .fp64) || values.any (fun p => p.1.toString == name)) do
          throwError "every integer runtime scalar needs an explicit call binding"
      -- A pointer argument may already point into an allocation. This
      -- binds that call value without changing the pasted callee body. Offset
      -- units are elements in the existing region model; native byte/width
      -- and allocation validity remain the caller's existing obligations.
      if let some offsets := pointerOffsets then
        let mut pointerValues : List (Name × TSyntax `tritonExpr) := []
        for arg in offsets do
          let `(tritonCallBinding| $id:ident := $value:term) := arg
            | throwErrorAt arg "invalid pointer offset binding"
          let name := id.getId.eraseMacroScopes
          unless pointers.any (fun p => p.1 == name.toString) do
            throwErrorAt id "offset binding is not a pointer parameter"
          if pointerValues.any (fun p => p.1 == name) then
            throwErrorAt id "duplicate pointer offset binding"
          let argument ← elabTerm value none
          let carrier ← inferType argument
          let expression ←
            if ← isDefEq carrier (mkConst ``Nat) then
              liftMacroM `(tritonExpr| ($id:ident + $(($value:term : Nat))))
            else if ← isDefEq carrier (mkConst ``Int) then
              liftMacroM `(tritonExpr| ($id:ident + $(($value:term : Int))))
            else throwErrorAt value "pointer offset must have a Nat or Int element-offset carrier"
          pointerValues := pointerValues ++ [(name,expression)]
        values := values ++ pointerValues
      -- Validate call bindings before elaborating constexpr values, so an
      -- incomplete call reports its missing argument rather than a later
      -- undefined constexpr. Resolve string None tests before pointer tests.
      let (stmts, optionalStrings) ← OptionalString.specialize constants stmts
      for stmt in stmts do
        rejectArgumentMutation optionalStrings stmt.raw "optional string constexpr mutation is unsupported"
      let stmts ← liftMacroM <| OptionalPointer.specialize presence stmts
      let stmts ← liftMacroM <| stmts.mapM fun stmt => do
        return (⟨← PythonBody.markWeakLiterals stmt.raw⟩ : TSyntax `tritonStmt)
      -- Static numeric constants remain weak Python values. Reject later
      -- mutation before lowering the supported exact powers to fixed literals.
      let mut localConstants : List Name := []
      for stmt in stmts do
        rejectArgumentMutation localConstants stmt.raw
        if let `(tritonStmt| $id:ident : tl.constexpr = $_:tritonExpr) := stmt then
          localConstants := id.getId.eraseMacroScopes :: localConstants
      let stmts ← liftMacroM <| Constexpr.lower stmts.toList
      let stmts := stmts.toArray
      let assigned := stmts.toList.flatMap (fun st => assignedNames st.raw)
      let mut strings : StringChoices := []
      for constant in constants do
        let name := Name.mkSimple constant
        let value ← elabTerm (mkIdent name) none
        -- Classification must not solve an untyped numeral's carrier as
        -- String. The existing Nat/Int/Real specialization owns that choice.
        if (← whnf (← inferType value)).isConstOf ``String then
          for st in stmts do
            rejectArgumentMutation [name] st.raw "string constexpr mutation is unsupported"
          strings := strings ++ [(name, ← finiteStringValues value)]
      let stmts := (← liftMacroM <| specializeStringBlock strings stmts.toList).toArray
      for stmt in stmts do rejectConstexprIteratorShadow constants stmt.raw
      let stmts ← stmts.mapM fun st => do
        return (⟨← foldStaticNatQuotients (constants.map Name.mkSimple) assigned st.raw⟩ : TSyntax `tritonStmt)
      -- A constexpr float is a weak Python scalar, not a runtime FP32
      -- argument. Its surrounding tensor operation selects its precision.
      -- Mutable constexpr names remain local statements, never substituted.
      for constant in constants do
        let name := Name.mkSimple constant
        if assigned.contains name || optionalStrings.contains name then continue
        let id : TSyntax `term := ⟨mkIdent name⟩
        let value ← elabTerm id none
        let dtype ← inferType value
        -- String comparisons are compile-time Boolean expressions in the DSL.
        -- Keep the caller's String name visible to that existing compiler;
        -- finite branch pruning above has already checked its possible values.
        if (← whnf dtype).isConstOf ``String then continue
        let expression ←
          if ← isDefEq dtype (mkConst ``Nat) then
            -- Expose actual closed constexpr literals to the source-width
            -- prepass, which validates legal arange sizes and integer promotion.
            if let some n ← getNatValue? (← whnf value) then
              let literal : TSyntax `num := ⟨Syntax.mkNumLit (toString n)⟩
              liftMacroM `(tritonExpr| $(($literal:num : Nat)))
            else if let some checked ← checkedArangeTerm? value then
              liftMacroM `(tritonExpr| $($checked:term))
            else liftMacroM `(tritonExpr| $(($id:term : Nat)))
          else if ← isDefEq dtype (mkConst ``Int) then
            liftMacroM `(tritonExpr| $(($id:term : Int)))
          else if ← isDefEq dtype (mkConst ``Real) then
            liftMacroM `(tritonExpr| $(($id:term : ℝ)))
          else if ← isDefEq dtype (mkConst ``Bool) then
            liftMacroM `(tritonExpr| $(($id:term : Bool)))
          else throwError "unsupported constexpr carrier; expected Nat, Int, Real, Bool or a finite String"
        values := values ++ [(name,expression)]
      let integerArguments := scalars.filterMap fun (name,kind) =>
        match kind with
        | .integer | .i32 | .i64 | .u32 | .u64 => some (Name.mkSimple name)
        | _ => none
      -- A straight-line pointer assignment consumes its caller offset on the
      -- RHS exactly once. Later reads use the resulting local pointer register.
      -- Conditional, tuple and augmented first assignments still fail closed.
      let rebindable := integerArguments ++ pointers.map (fun p => Name.mkSimple p.1)
      let stmts ← bindCallStatements rebindable values stmts
      let stmts ← specializeImmutableRangeStrides stmts
      let stmts ← RankOneBlockPtr.lower stmts
      let stmts ← liftMacroM <| stmts.mapM fun st => do
        return (⟨← normalizeBoundaryTuples st.raw⟩ : TSyntax `tritonStmt)
      let metadata ← liftMacroM <| stmts.toList.mapM fun st => do
        return (⟨← sourcePortView st.raw⟩ : TSyntax `tritonStmt)
      let (reads, writes) := Metadata.blockRegions metadata
      for region in reads ++ writes do
        let `(term| $id:ident) := region
          | throwErrorAt region "call signature requires named external pointer parameters"
        unless pointers.any (fun p => p.1 == id.getId.eraseMacroScopes.toString) do
          throwErrorAt region "pointer parameter {id} is missing from the call signature"
      for (regions,ports) in [(reads,readPorts),(writes,writePorts)] do
        unless ports.isEmpty do
          for region in regions do
            let `(term| $id:ident) := region | throwErrorAt region "expected a pointer parameter"
            unless ports.contains id.getId.eraseMacroScopes.toString do
              throwErrorAt region "observed memory access is missing from the declared call ports"
      let constants := constants.map Name.mkSimple
      let term ← liftMacroM do
        let mut declarations : Array (TSyntax `tritonStmt) := #[]
        -- Invocation ports follow the declared Python parameter order even
        -- when the first load happens to read a later argument.
        let orderedPorts (regions : List (TSyntax `term)) : MacroM (Array (TSyntax `term)) := do
          let mut result := #[]
          for (name, _) in pointers do
            for region in regions do
              if let `(term| $id:ident) := region then
                if id.getId.eraseMacroScopes.toString == name then
                  result := result.push region
                  break
          return result
        let explicitPorts (names : List String) : Array (TSyntax `term) :=
          (names.map (fun name => (⟨mkIdent (Name.mkSimple name)⟩ : TSyntax `term))).toArray
        let inputs ← if readPorts.isEmpty then orderedPorts reads else pure (explicitPorts readPorts)
        let outputs ← if writePorts.isEmpty then orderedPorts writes else pure (explicitPorts writePorts)
        declarations := declarations.push (← `(tritonStmt| tl.ports inputs=[$inputs,*], outputs=[$outputs,*]))
        for (name, kind) in pointers do
          let id := mkIdent (Name.mkSimple name)
          if naturalIntegerNames.contains name then
            -- The existing unsigned storage spelling selects the Nat memory
            -- carrier in the expression compiler. Source signedness/width
            -- remains the declared PointerType above; this mathematical view
            -- introduces neither an instruction nor an unsigned native ABI.
            -- An explicit source .to(int64) must cast this load, never change
            -- its underlying memory channel through later address-use hints.
            declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = tl.uint64))
            continue
          let dt ← if (typedFP32 || typedFP32Names.contains name) && kind == .fp32 then
              `(tritonDType| DTYPE)
            else do
              let some dt ← dtypeSyntaxOfName? (PointerType.dtypeName kind)
                | Macro.throwError "unsupported pointer dtype"
              pure dt
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = $dt:tritonDType))
        for (name, kind) in scalars do
          if values.any (fun p => p.1.toString == name) then continue
          let id := mkIdent (Name.mkSimple name)
          let dtName := match kind with
            | .fp32 => `tl.float32 | .fp64 => `tl.float64
            | .integer | .i64 => `tl.int64 | .i32 => `tl.int32
            | .u32 => `tl.uint32 | .u64 => `tl.uint64
          let some dt ← dtypeSyntaxOfName? dtName
            | Macro.throwError "unsupported scalar dtype"
          declarations := declarations.push (← `(tritonStmt| tl.param $id:ident : $dt:tritonDType))
        let scalarTypes := scalars.filterMap fun (name,kind) =>
          if values.any (fun p => p.1.toString == name) then none else
            match kind with
            | .fp32 => some (Name.mkSimple name, TypedStores.ValueInfo.typed .fp32)
            | .fp64 => some (Name.mkSimple name, TypedStores.ValueInfo.typed .fp64)
            | _ => none
        let floatingPointers := pointers.filterMap fun (name,kind) =>
          if [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.inputFloat,.outputFloat].contains kind
            then some (Name.mkSimple name) else none
        let stmts ← NestedLoadAssignments.lower floatingPointers stmts
        let body ← TypedStores.lower (pointers.map (fun (name, kind) => (Name.mkSimple name, kind))) scalarTypes stmts.toList
        let storagePointers := pointers.filterMap fun (name,kind) =>
          if kind == .fp32 && (typedFP32 || typedFP32Names.contains name)
          then some (Name.mkSimple name) else none
        let body ← TypedStorageReduction.lower storagePointers body
        let emitted := finiteDeclarations ++ declarations ++ body.toArray
        let compiled ← `(triton { $emitted* })
        if usesTypedFP32 then
          -- DTYPE is interface metadata, evaluated before any source statement.
          let dtype := mkIdent `dtype
          `(let $dtype:ident : FloatDType := FloatDType.fp32; $compiled)
        else pure compiled
      let term : TSyntax `term := ⟨← specializeRanges constants assigned term.raw⟩
      let result ← elabTerm term none
      if fp32Values then
        synthesizeSyntheticMVarsNoPostponing
        let checked ← instantiateMVars result
        checkFP32CallPrecision checked
      let result ← instantiateMVars result
      -- ABI provenance has already selected the checked RNG source profile.
      -- Erase these defined identity adapters from published values, just as
      -- with the checked FP32 precision marker. The RNG width/domain stays IR.
      return result.replace fun e =>
        if e.isAppOfArity ``TypedStores.fp32CallValue 1 then some e.appArg! else
        if e.isAppOfArity ``Random.sourceIntCallValue 2 ||
            e.isAppOfArity ``Random.sourceNatCallValue 2 then some e.appArg! else none

/-- A caller's Boolean constexpr may select differently shaped masks or None.
Elaborate every possible specialization and retain the caller's actual Boolean
selector. Runtime Boolean parameters never select this path. -/
private def elaborateCall (profile : TSyntax `term) (stmts : Array (TSyntax `tritonStmt))
    (callValues : Option (Array (TSyntax `tritonCallBinding)))
    (pointerOffsets : Option (Array (TSyntax `tritonCallBinding)) := none)
    (pointerPresence : Option (Array (TSyntax `tritonCallBinding)) := none) : TermElabM Expr := do
  let signature ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
  let selected ← (← listValues (← mkAppM ``Triton.CallSignature.specializeBooleans #[signature])).mapM stringValue
  if !selected.isEmpty then
    for stmt in stmts do rejectBodyMetadata stmt.raw
    let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[signature])).mapM stringValue
    unless selected.eraseDups.length == selected.length && selected.all constants.contains do
      throwError "Boolean specialization names must be distinct constexpr parameters"
    let mut choices : ConstexprMask.Choices := []
    for name in selected do
      let id := Name.mkSimple name
      let value ← elabTermEnsuringType (mkIdent id) (mkConst ``Bool)
      let reduced ← whnf value
      unless reduced.isConstOf ``Bool.true || reduced.isConstOf ``Bool.false do
        throwError "explicit Boolean specialization requires a closed caller value"
      for stmt in stmts do
        rejectArgumentMutation [id] stmt.raw "specialized Boolean constexpr cannot be rebound"
      choices := choices ++ [(id,reduced.isConstOf ``Bool.true)]
    let body ← liftMacroM <| stmts.mapM fun st => do
      return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
    let body ← liftMacroM <| ConstexprMask.lower choices body
    return ← elaborateCallCore profile body callValues pointerOffsets pointerPresence
  unless stmts.any (fun st => ConstexprMask.needed st.raw) do
    return ← elaborateCallCore profile stmts callValues pointerOffsets pointerPresence
  for stmt in stmts do rejectBodyMetadata stmt.raw
  let signature ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
  let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[signature])).mapM stringValue
  let mut booleans : List (Name × Expr) := []
  for name in constants do
    let id := Name.mkSimple name
    let value ← elabTerm (mkIdent id) none
    if (← whnf (← inferType value)).isConstOf ``Bool then
      for stmt in stmts do
        rejectArgumentMutation [id] stmt.raw "Boolean constexpr cannot be rebound during mask specialization"
      booleans := booleans ++ [(id,value)]
  let stmts ← liftMacroM <| stmts.mapM fun st => do
    return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
  let rec cases (remaining : List (Name × Expr)) (choices : ConstexprMask.Choices) : TermElabM Expr := do
    match remaining with
    | [] =>
        let body ← liftMacroM <| ConstexprMask.lower choices stmts
        elaborateCallCore profile body callValues pointerOffsets pointerPresence
    | (name,value) :: rest =>
        let reduced ← whnf value
        if reduced.isConstOf ``Bool.false then return ← cases rest ((name,Bool.false) :: choices)
        if reduced.isConstOf ``Bool.true then return ← cases rest ((name,Bool.true) :: choices)
        let no ← cases rest ((name,Bool.false) :: choices)
        let yes ← cases rest ((name,Bool.true) :: choices)
        let motive := mkLambda `flag BinderInfo.default (mkConst ``Bool) (mkConst ``Triton.ComputeKernel)
        mkAppOptM ``Bool.casesOn #[some motive,some value,some no,some yes]
  cases booleans []

/-- The call provides the dtype; the copied function retains dtype queries.
Each finite dtype case goes through the same ordinary checked call elaborator. -/
private def elaborateSourceDType (profile : TSyntax `term) (pointer : Ident)
    (kind : TSyntax `term) (stmts : Array (TSyntax `tritonStmt)) : TermElabM Expr := do
  let signature ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
  let declared ← (← listValues (← mkAppM ``Triton.CallSignature.sourceDTypePointers #[signature])).mapM stringValue
  let name := pointer.getId.eraseMacroScopes.toString
  unless declared == [name] do
    throwError "source_dtype requires exactly the declared source dtype pointer"
  let pointers ← bindings dtypeValue (← mkAppM ``Triton.CallSignature.pointers #[signature])
  unless pointers.all (fun p => p.1 != name) do
    throwError "source dtype pointer cannot also have a fixed pointer dtype"
  for stmt in stmts do rejectBodyMetadata stmt.raw
  let actual ← elabTermEnsuringType kind (mkConst ``Triton.SourceDType)
  let cases : List (Triton.SourceDType × Name) :=
    [(.fp16,``PointerType.fp16),(.bf16,``PointerType.bf16),(.fp32,``PointerType.fp32),
     (.fp64,``PointerType.fp64),(.f8e4,``PointerType.f8e4),(.f8e5,``PointerType.f8e5),
     (.i8,``PointerType.i8),(.i16,``PointerType.i16),(.i32,``PointerType.i32),(.i64,``PointerType.i64),
     (.u8,``PointerType.u8),(.u16,``PointerType.u16),(.u32,``PointerType.u32),(.u64,``PointerType.u64),
     (.boolean,``PointerType.boolean)]
  let mut branches : Array Expr := #[]
  for (sourceType,storageType) in cases do
    let label := Syntax.mkStrLit name
    let storage := mkIdent storageType
    let specialized ← `(({ $profile:term with sourceDTypePointers := [], pointers := [($label:str,$storage:ident)] ++ ($profile:term).pointers } : Triton.CallSignature))
    let body ← liftMacroM <| SourceDType.specializeBody pointer.getId sourceType stmts
    branches := branches.push (← elaborateCall specialized body none)
  let motive := mkLambda `sourceDType BinderInfo.default (mkConst ``Triton.SourceDType)
    (mkConst ``Triton.ComputeKernel)
  mkAppOptM ``Triton.SourceDType.casesOn (#[some motive,some actual] ++ branches.map some)

elab_rules : term
  | `(with_triton_signature $profile:term pointer_presence [$presence:tritonCallBinding,*] in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts none none (some presence.getElems)
  | `(with_triton_call $profile:term call_args [$args:tritonCallBinding,*]
      pointer_presence [$presence:tritonCallBinding,*] in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts (some args.getElems) none (some presence.getElems)
  | `(with_triton_signature $profile:term source_dtype $pointer:ident := $kind:term in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateSourceDType profile pointer kind stmts
  | `(with_triton_signature $profile:term in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts none
  | `(with_triton_call $profile:term call_args [$args:tritonCallBinding,*] in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts (some args.getElems)
  | `(with_triton_call $profile:term call_args [$args:tritonCallBinding,*]
      pointer_offsets [$offsets:tritonCallBinding,*] in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts (some args.getElems) (some offsets.getElems)

end DSL.CallSignature

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

end VeriTile.Triton
