import VeriTile.Triton.DSL.TypedStores
import VeriTile.Triton.DSL.PythonBody
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

/-- A per-call size has already been supplied before executing the body.
Select the existing constant-bound range constructor only for known immutable
Nat bounds. Runtime locals and signed casts retain their dynamic range path. -/
private partial def specializeRanges (constants assigned : List Name) (raw : Syntax) : TermElabM Syntax := do
  let args ← raw.getArgs.mapM (specializeRanges constants assigned)
  let next := match raw with | .node info kind _ => Syntax.node info kind args | _ => raw
  if let `(tritonStmt| for $i:ident in range($n:tritonExpr) { $body:tritonStmt* }) := next then
    if let some n ← staticNatBound constants assigned n then
      return (← liftMacroM `(tritonStmt| tl.for $i in $($n) { $body* })).raw
  if let `(tritonStmt| for $i:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $body:tritonStmt* }) := next then
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

/-- A verification call specializes immutable runtime size values supplied by
the caller. Their source ABI remains in `scalars`, never in `constexpr`.
Nat/Int carriers use the existing mathematical index projection;
native signed-width bounds must still be established by the task's IO/contract. -/
private def elaborateCall (profile : TSyntax `term) (stmts : Array (TSyntax `tritonStmt))
    (callValues : Option (Array (TSyntax `tritonCallBinding)))
    (pointerOffsets : Option (Array (TSyntax `tritonCallBinding)) := none) : TermElabM Expr := do
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
            | .i32 | .i64 | .integer => do
                if ← isDefEq dtype (mkConst ``Nat) then
                  liftMacroM `(tritonExpr| $(($value:term : Nat)))
                else if ← isDefEq dtype (mkConst ``Int) then
                  liftMacroM `(tritonExpr| $(($value:term : Int)))
                else throwErrorAt value "per-call integer value must have a Nat or Int carrier"
            | .fp64 => throwErrorAt value "per-call FP64 literal specialization is unsupported; use runtime scalar binding"
          values := values ++ [(name, expression)]
        unless scalars.all (fun (name,kind) =>
            (kind matches .fp32 | .fp64) || values.any (fun p => p.1.toString == name)) do
          throwError "every integer runtime scalar needs an explicit call binding"
        for stmt in stmts do rejectArgumentMutation (values.map Prod.fst) stmt.raw
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
          unless ← isDefEq (← inferType argument) (mkConst ``Nat) do
            throwErrorAt value "pointer offset must have a Nat element-offset carrier"
          let expression ← liftMacroM `(tritonExpr| ($id:ident + $(($value:term : Nat))))
          pointerValues := pointerValues ++ [(name,expression)]
        for stmt in stmts do
          rejectArgumentMutation (pointerValues.map Prod.fst) stmt.raw
            "pointer-offset specialization does not yet support reassigned pointer arguments"
        values := values ++ pointerValues
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
      for stmt in stmts do rejectConstexprIteratorShadow constants stmt.raw
      -- A constexpr float is a weak Python scalar, not a runtime FP32
      -- argument. Its surrounding tensor operation selects its precision.
      -- Mutable constexpr names remain local statements, never substituted.
      for constant in constants do
        let name := Name.mkSimple constant
        if assigned.contains name then continue
        let id : TSyntax `term := ⟨mkIdent name⟩
        let value ← elabTerm id none
        let dtype ← inferType value
        let expression ←
          if ← isDefEq dtype (mkConst ``Nat) then
            liftMacroM `(tritonExpr| $(($id:term : Nat)))
          else if ← isDefEq dtype (mkConst ``Int) then
            liftMacroM `(tritonExpr| $(($id:term : Int)))
          else if ← isDefEq dtype (mkConst ``Real) then
            liftMacroM `(tritonExpr| $(($id:term : ℝ)))
          else if ← isDefEq dtype (mkConst ``Bool) then
            liftMacroM `(tritonExpr| $(($id:term : Bool)))
          else throwError "unsupported constexpr carrier; expected Nat, Int, Real or Bool"
        values := values ++ [(name,expression)]
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
      let term : TSyntax `term := ⟨← specializeRanges constants assigned term.raw⟩
      let result ← elabTerm term none
      if fp32Values then
        synthesizeSyntheticMVarsNoPostponing
        let result ← instantiateMVars result
        checkFP32CallPrecision result
        -- Erase checked, definitionally identity provenance before publishing IR.
        return result.replace fun e =>
          if e.isAppOfArity ``TypedStores.fp32CallValue 1 then some e.appArg! else none
      return result

elab_rules : term
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
