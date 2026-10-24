import VeriTile.Triton.DSL.TypedStores
import VeriTile.Triton.DSL.PythonBody
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
  | i32 | i64

structure CallSignature where
  pointers : List (String × SourceDType) := []
  scalars : List (String × ScalarType) := []
  constexpr : List String := []
  /-- Optional public memory roles when pointer rebinding prevents the legacy
  collector from recovering the original region. These are interface metadata,
  not instructions or claims that an access succeeds. -/
  readPorts : List String := []
  writePorts : List String := []

namespace DSL.CallSignature
open Lean Elab Term Meta

syntax "with_triton_signature " term " in " "triton" tritonPythonBody : term
declare_syntax_cat tritonCallBinding
syntax ident " := " term : tritonCallBinding
syntax "with_triton_call " term " call_args " "[" tritonCallBinding,* "]" " in "
  "triton" tritonPythonBody : term

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

/-- A verification call specializes immutable runtime size values supplied by
the caller. Their source ABI remains in `scalars`, never in `constexpr`.
Nat/Int carriers use the existing mathematical index projection;
native signed-width bounds must still be established by the task's IO/contract. -/
private def elaborateCall (profile : TSyntax `term) (stmts : Array (TSyntax `tritonStmt))
    (callValues : Option (Array (TSyntax `tritonCallBinding))) : TermElabM Expr := do
      let profile ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
      let pointers ← bindings dtypeValue (← mkAppM ``Triton.CallSignature.pointers #[profile])
      let scalars ← bindings scalarTypeValue (← mkAppM ``Triton.CallSignature.scalars #[profile])
      let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM stringValue
      let readPorts ← (← listValues (← mkAppM ``Triton.CallSignature.readPorts #[profile])).mapM stringValue
      let writePorts ← (← listValues (← mkAppM ``Triton.CallSignature.writePorts #[profile])).mapM stringValue
      for ports in [readPorts,writePorts] do
        unless ports.eraseDups.length == ports.length && ports.all (fun name => pointers.any (fun p => p.1 == name)) do
          throwError "memory ports must be distinct pointer parameter names"
      let names := pointers.map Prod.fst ++ scalars.map Prod.fst ++ constants
      unless names.eraseDups.length == names.length do throwError "duplicate name in call signature"
      for stmt in stmts do rejectBodyMetadata stmt.raw
      let mut values : List (Name × TSyntax `tritonExpr) := []
      if let some args := callValues then
        for arg in args do
          let `(tritonCallBinding| $id:ident := $value:term) := arg
            | throwErrorAt arg "invalid call argument binding"
          let name := id.getId.eraseMacroScopes
          if values.any (fun p => p.1 == name) then throwErrorAt arg "duplicate call argument binding"
          let some (_, kind) := scalars.find? (fun p => p.1 == name.toString)
            | throwErrorAt arg "binding is not a runtime scalar parameter"
          unless kind matches .i32 | .i64 | .integer do
            throwErrorAt arg "per-call specialization requires an integer source argument"
          let argument ← elabTerm value none
          let dtype ← inferType argument
          let expression ← if ← isDefEq dtype (mkConst ``Nat) then
              liftMacroM `(tritonExpr| $(($value:term : Nat)))
            else if ← isDefEq dtype (mkConst ``Int) then
              liftMacroM `(tritonExpr| $(($value:term : Int)))
            else throwErrorAt value "per-call integer value must have a Nat or Int carrier"
          values := values ++ [(name, expression)]
        unless scalars.all (fun (name,kind) =>
            (kind matches .fp32 | .fp64) || values.any (fun p => p.1.toString == name)) do
          throwError "every integer runtime scalar needs an explicit call binding"
        for stmt in stmts do rejectArgumentMutation (values.map Prod.fst) stmt.raw
      let stmts ← liftMacroM <| stmts.mapM fun st => do
        return (⟨← bindSourceValues values st.raw⟩ : TSyntax `tritonStmt)
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
          let some dt ← dtypeSyntaxOfName? (SourceDType.dtypeName kind)
            | Macro.throwError "unsupported pointer dtype"
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = $dt:tritonDType))
        for (name, kind) in scalars do
          if values.any (fun p => p.1.toString == name) then continue
          let id := mkIdent (Name.mkSimple name)
          let dtName := match kind with
            | .fp32 => `tl.float32 | .fp64 => `tl.float64
            | .integer | .i64 => `tl.int64 | .i32 => `tl.int32
          let some dt ← dtypeSyntaxOfName? dtName
            | Macro.throwError "unsupported scalar dtype"
          declarations := declarations.push (← `(tritonStmt| tl.param $id:ident : $dt:tritonDType))
        let scalarTypes := scalars.filterMap fun (name,kind) =>
          if values.any (fun p => p.1.toString == name) then none else
            match kind with
            | .fp32 => some (Name.mkSimple name, TypedStores.ValueInfo.typed .fp32)
            | .fp64 => some (Name.mkSimple name, TypedStores.ValueInfo.typed .fp64)
            | _ => none
        let body ← TypedStores.lower (pointers.map (fun (name, kind) => (Name.mkSimple name, kind))) scalarTypes stmts.toList
        let emitted := declarations ++ body.toArray
        `(triton { $emitted* })
      elabTerm term none

elab_rules : term
  | `(with_triton_signature $profile:term in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts none
  | `(with_triton_call $profile:term call_args [$args:tritonCallBinding,*] in triton $body:tritonPythonBody) => do
      let environment ← getEnv
      let stmts ← liftMacroM <| PythonBody.statements environment body
      elaborateCall profile stmts (some args.getElems)

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
