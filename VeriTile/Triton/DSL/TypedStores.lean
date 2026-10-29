import VeriTile.Triton.DSL.NestedLoadAssignments
import VeriTile.Triton.DSL.CallValue
import VeriTile.Triton.DSL.PointerType
import VeriTile.Triton.DSL.PythonConstants

/-! Call-site store conversion with conservative source dtype tracking.
A load and a register copy retain the source pointer's element type. Storing
that value into the same element type is an identity, with no new cast event.
Unknown value types still receive the destination conversion. -/
namespace VeriTile.Triton.DSL.TypedStores
open Lean

abbrev Types := List (Name × Triton.PointerType)

inductive ValueInfo where
  | typed (dtype : Triton.PointerType)
  | integerLiteral | integerScalar | floatingLiteral
  deriving BEq
abbrev Values := List (Name × ValueInfo)

private def name (id : Ident) : Name := id.getId.eraseMacroScopes

/-- A sequential IR tuple is valid only while every RHS still observes its
entry values. A later RHS that reads an earlier target would require register
capture; reject it instead of silently changing Python evaluation order. -/
private partial def readsTupleTarget (target : Name) (raw : Syntax) : Bool :=
  -- Caller-owned Lean terms are immutable values, not source register reads.
  if let `(tritonExpr| $($_:term)) := raw then Bool.false else
  if raw.isIdent then
    let value := raw.getId.eraseMacroScopes.toString
    value == target.toString || value.startsWith (target.toString ++ ".")
  else raw.getArgs.any (readsTupleTarget target)

private def lookup (types : Types) (id : Ident) : Option Triton.PointerType :=
  (types.find? (fun entry => entry.1 == name id)).map Prod.snd

/-- Block-pointer construction and advance preserve their base element type.
The expression compiler separately validates every shape/stride/order field. -/
private def blockPointerBase (e : TSyntax `tritonExpr) : Option (TSyntax `tritonExpr) := do
  let args := e.raw.getArgs
  let first ← args[0]?
  unless first.isAtom do failure
  if first.getAtomVal == "tl.make_block_ptr(" then
    if args[1]?.any Syntax.isIdent && args[2]?.any (fun a => a.isAtom && a.getAtomVal == "=") then
      unless args[1]!.getId.eraseMacroScopes == `base do failure
      return ⟨← args[3]?⟩
    return ⟨← args[1]?⟩
  if first.getAtomVal == "tl.advance(" then return ⟨← args[1]?⟩
  failure

private partial def pointerType (types : Types) (e : TSyntax `tritonExpr) : Option Triton.PointerType :=
  match e with
  | `(tritonExpr| $id:ident) => lookup types id
  | `(tritonExpr| ($inner:tritonExpr)) => pointerType types inner
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      match pointerType types a, pointerType types b with
      | some dtype, none | none, some dtype => some dtype
      | _, _ => none
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      if (pointerType types b).isSome then none else pointerType types a
  | _ => do
      let base ← blockPointerBase e
      pointerType types base

private partial def dtypeOverride (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal == "dtype") ||
  (raw.isIdent && raw.getId.eraseMacroScopes == `dtype) || raw.getArgs.any dtypeOverride

private def sourceKinds : List Triton.PointerType :=
  [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean,.inputFloat,.outputFloat]

private partial def dtypeToken (raw : Syntax) : Option Name :=
  if raw.isAtom then
    (sourceKinds.find? (fun dtype => (PointerType.dtypeName dtype).toString == raw.getAtomVal)).map PointerType.dtypeName
  else if raw.isIdent then some raw.getId.eraseMacroScopes
  else if raw.getNumArgs == 1 then dtypeToken raw[0]
  else none

private def explicitCastType (pointers : Types) (values : Values) (e : TSyntax `tritonExpr) : Option Triton.PointerType := do
  let direct : Option (TSyntax `tritonDType) := match e with
    | `(tritonExpr| tl.cast($_:tritonExpr, $dtype:tritonDType)) => some dtype
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($dtype:tritonDType)) => some dtype
    | _ => none
  if let some dt := direct then
    let target ← dtypeToken dt.raw
    return ← sourceKinds.find? (fun dtype => PointerType.dtypeName dtype == target)
  unless (methodCast? e).isSome ||
      [``tritonMethodCastElementTy,``tritonMethodCastElementTyIdent,
       ``tritonIdentMethodCastElementTyIdent,``tritonIdentMethodCastElementTyIdentSpaced].contains e.raw.getKind ||
      e.raw.getArgs.any (fun arg => arg.isAtom && arg.getAtomVal == "to") do failure
  let target ← (methodCastTargetName? e).orElse fun _ => do
    let (_, dtype) ← methodCast? e
    dtypeToken dtype.raw
  match target with
  | .str (.str pointer "dtype") "element_ty"
  | .str (.str pointer "type") "element_ty" => lookup pointers (mkIdent pointer)
  | .str tensor "dtype" => do
      let (_, .typed dtype) ← values.find? (fun entry => entry.1 == tensor) | none
      return dtype
  | _ => sourceKinds.find? (fun dtype => PointerType.dtypeName dtype == target)

private def constructorType (pointers : Types) (e : TSyntax `tritonExpr) : Option Triton.PointerType := do
  let resolve (target : Name) : Option Triton.PointerType :=
    match target with
    | .str (.str pointer "dtype") "element_ty"
    | .str (.str pointer "type") "element_ty" => lookup pointers (mkIdent pointer)
    | _ => sourceKinds.find? (fun dtype => PointerType.dtypeName dtype == target)
  let args := e.raw.getArgs
  let first ← args[0]?
  unless first.isAtom && ["tl.zeros(","tl.full(","tl.sum(","tl.cumsum("].contains first.getAtomVal do failure
  for i in [:args.size] do
    if args[i]!.isIdent && args[i]!.getId.eraseMacroScopes == `dtype && i+2 < args.size then
      let target ← dtypeToken args[i+2]!
      return ← resolve target
  for arg in args do
    if let some target := dtypeToken arg then
      if let some dtype := resolve target then
        return dtype
  match e with
  | `(tritonExpr| tl.zeros([$_:tritonExpr,*])) => return .fp32
  | _ => failure

private partial def valueType (pointers : Types) (values : Values)
    (e : TSyntax `tritonExpr) : Option ValueInfo :=
  let sameWorkingFloat (a b : TSyntax `tritonExpr) : Option Triton.PointerType := do
    let left ← valueType pointers values a
    let right ← valueType pointers values b
    match left, right with
    | .typed .fp64, .integerLiteral | .integerLiteral, .typed .fp64
    | .typed .fp64, .integerScalar | .integerScalar, .typed .fp64
    | .typed .fp64, .floatingLiteral | .floatingLiteral, .typed .fp64 => some .fp64
    | .typed .fp64, .typed other | .typed other, .typed .fp64 =>
        if [.fp64,.fp32,.fp16,.bf16,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains other
        then some .fp64 else none
    | .typed .fp32, .typed .fp32
    | .typed .fp32, .integerLiteral | .integerLiteral, .typed .fp32
    | .typed .fp32, .integerScalar | .integerScalar, .typed .fp32
    | .typed .fp32, .floatingLiteral | .floatingLiteral, .typed .fp32 => some .fp32
    | .typed .fp32, .typed integer | .typed integer, .typed .fp32 =>
        if [.fp16,.bf16,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains integer then some .fp32 else none
    | .typed .fp16, .integerLiteral | .integerLiteral, .typed .fp16
    | .typed .fp16, .integerScalar | .integerScalar, .typed .fp16
    | .typed .fp16, .floatingLiteral | .floatingLiteral, .typed .fp16 => some .fp16
    | .typed .bf16, .integerLiteral | .integerLiteral, .typed .bf16
    | .typed .bf16, .floatingLiteral | .floatingLiteral, .typed .bf16 => some .bf16
    | .typed left, .typed right =>
        if left == right && [.fp16,.bf16].contains left then some left
        else if (left == .fp16 && right == .bf16) || (left == .bf16 && right == .fp16) then some .fp16
        else if [.fp16,.bf16].contains left &&
            [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains right then some (if left == .bf16 then .fp32 else left)
        else if [.fp16,.bf16].contains right &&
            [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains left then some (if right == .bf16 then .fp32 else right)
        else none
    | _, _ => none
  let dotInputs (a b : TSyntax `tritonExpr) : Bool :=
    match valueType pointers values a, valueType pointers values b with
    | some (.typed left), some (.typed right) =>
        (left == right && [.fp16,.bf16,.fp32].contains left) ||
          ([PointerType.f8e4,.f8e5].contains left && [.f8e4,.f8e5].contains right)
    | _, _ => Bool.false
  if let some dtype := (explicitCastType pointers values e).orElse (fun _ => constructorType pointers e) then some (.typed dtype) else
  match e with
  | `(tritonExpr| tl.__python_literal($inner:tritonExpr)) => valueType pointers values inner
  | `(tritonExpr| $id:ident) => (values.find? (fun entry => entry.1 == name id)).map Prod.snd
  | `(tritonExpr| ($inner:tritonExpr)) => valueType pointers values inner
  | `(tritonExpr| tl.broadcast_to($inner:tritonExpr, [$_:tritonExpr,*]))
  | `(tritonExpr| tl.reshape($inner:tritonExpr, [$_:tritonExpr,*])) =>
      valueType pointers values inner
  | `(tritonExpr| $((VeriTile.Triton.DSL.TypedStores.fp32CallValue $_:term : ℝ))) => some (.typed .fp32)
  | `(tritonExpr| $(($_:term : Nat))) | `(tritonExpr| $(($_:term : Int))) => some .integerScalar
  | `(tritonExpr| $(($_:term : ℝ))) => some .floatingLiteral
  | `(tritonExpr| $_:num) | `(tritonExpr| -$_:num) => some .integerLiteral
  | `(tritonExpr| $_:scientific) | `(tritonExpr| -$_:scientific)
  | `(tritonExpr| -inf)
  | `(tritonExpr| float($_:term)) | `(tritonExpr| -float($_:term)) => some .floatingLiteral
  | `(tritonExpr| tl.load($p:tritonExpr, $_mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else (pointerType pointers p).map ValueInfo.typed
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else (pointerType pointers p).map ValueInfo.typed
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) =>
      ((sameWorkingFloat a b).map fun dtype =>
        if [.fp16,.bf16].contains dtype then .fp32 else dtype).map ValueInfo.typed
  | `(tritonExpr| tl.div_rn($_:tritonExpr, $_:tritonExpr)) => some (.typed .fp32)
  | `(tritonExpr| tl.where($_:tritonExpr, $a:tritonExpr, $b:tritonExpr)) =>
      (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr))
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) =>
      (sameWorkingFloat a b).map ValueInfo.typed
  | `(tritonExpr| $_:tritonExpr < $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr <= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr > $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr >= $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr == $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr != $_:tritonExpr) => some (.typed .boolean)
  | `(tritonExpr| tl.sin($a:tritonExpr)) | `(tritonExpr| tl.math.sin($a:tritonExpr))
  | `(tritonExpr| tl.cos($a:tritonExpr)) | `(tritonExpr| tl.exp($a:tritonExpr))
  | `(tritonExpr| tl.exp2($a:tritonExpr)) | `(tritonExpr| tl.math.exp2($a:tritonExpr))
  | `(tritonExpr| tl.sigmoid($a:tritonExpr)) | `(tritonExpr| tl.rsqrt($a:tritonExpr))
  | `(tritonExpr| tl.math.rsqrt($a:tritonExpr))
  | `(tritonExpr| libdevice.rsqrt($a:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.rsqrt($a:tritonExpr))
  | `(tritonExpr| libdevice.exp($a:tritonExpr))
  | `(tritonExpr| libdevice.tanh($a:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.tanh($a:tritonExpr))
  | `(tritonExpr| tl.log($a:tritonExpr)) | `(tritonExpr| tl.sqrt($a:tritonExpr)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($a:tritonExpr, $_:num)) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| tl.sum($a:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      -- Triton 3.7.1 keeps the floating input dtype unless dtype= overrides it.
      -- An unresolved override must not fall back to the input's type.
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else
      match valueType pointers values a with
      | some (.typed dtype) =>
          if [.fp16,.bf16,.fp32,.fp64].contains dtype then some (.typed dtype) else none
      | _ => none
  | `(tritonExpr| tl.cumsum($a:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
      if kwargs.any (fun kw => dtypeOverride kw.raw) then none else
      match valueType pointers values a with
      | some (.typed dtype) =>
          if [.fp16,.fp32,.fp64].contains dtype then some (.typed dtype) else none
      | _ => none
  | `(tritonExpr| tl.max($a:tritonExpr, $_axis:num)) =>
      -- Positional axes use the same value-only reduction promotion as
      -- keyword axes (Triton 3.7.1 standard.max).
      match valueType pointers values a with
      | some (.typed .fp16) | some (.typed .bf16) | some (.typed .fp32) => some (.typed .fp32)
      | some (.typed .fp64) => some (.typed .fp64)
      | _ => none
  | `(tritonExpr| tl.max($a:tritonExpr $[, $_:tritonReduceKwarg]*)) =>
      match valueType pointers values a with
      | some (.typed .fp16) | some (.typed .bf16) | some (.typed .fp32) => some (.typed .fp32)
      | some (.typed .fp64) => some (.typed .fp64)
      | _ => none
  | `(tritonExpr| -$a:tritonExpr) =>
      if valueType pointers values a == some (.typed .fp32) then some (.typed .fp32) else none
  | `(tritonExpr| $a:tritonExpr[:, None])
  | `(tritonExpr| $a:tritonExpr[None, :])
  | `(tritonExpr| $a:tritonExpr[None])
  | `(tritonExpr| $a:tritonExpr[:, None, None])
  | `(tritonExpr| $a:tritonExpr[None, :, None])
  | `(tritonExpr| $a:tritonExpr[None, None, :])
  | `(tritonExpr| $a:tritonExpr[:, None, :])
  | `(tritonExpr| $a:tritonExpr[None, :, :])
  | `(tritonExpr| $a:tritonExpr[:, :, None]) => valueType pointers values a
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr)) =>
      if dotInputs a b &&
          valueType pointers values acc == some (.typed .fp32)
      then some (.typed .fp32) else none
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      -- Triton 3.7.1 semantic.dot accepts matching fp16/bf16/fp32 inputs
      -- or any supported fp8 pair with an FP32 accumulator/out_dtype.
      -- The live accumulator is an operand, never a separate add.
      unless dotInputs a b &&
          valueType pointers values acc == some (.typed .fp32) do failure
      for kw in kwargs do
        if let `(tritonReduceKwarg| out_dtype=$dtype:tritonDType) := kw then
          unless (← dtypeToken dtype.raw) == PointerType.dtypeName .fp32 do failure
      return .typed .fp32
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      unless dotInputs a b do failure
      let fp32Inputs := valueType pointers values a == some (.typed .fp32) &&
        valueType pointers values b == some (.typed .fp32)
      -- The newly supported non-FP32 input path retains the default FP32
      -- result. Other explicit result formats need their own checked path.
      if !fp32Inputs then
        for kw in kwargs do
          if let `(tritonReduceKwarg| out_dtype=$dtype:tritonDType) := kw then
            unless (← dtypeToken dtype.raw) == PointerType.dtypeName .fp32 do failure
      let mut result := Triton.PointerType.fp32
      for kw in kwargs do
        if let `(tritonReduceKwarg| out_dtype=$dtype:tritonDType) := kw then
          let target ← dtypeToken dtype.raw
          result ← sourceKinds.find? (fun kind => PointerType.dtypeName kind == target)
      return .typed result
  | `(tritonExpr| tl.trans($a:tritonExpr)) => valueType pointers values a
  | _ => none

/-- An ordinary Python assignment materializes a numeric scalar as a tensor;
an inline scalar remains weak. Classify the source floating literal before it
enters register dataflow. Triton's to_tensor_type uses the FP32 normal range,
zero and infinities; other finite literals require an FP64 tensor frontend.
See triton/compiler/code_generator.py::_sanitize_value and
triton/language/semantic.py::to_tensor_type in the upstream implementation. -/
private partial def literalTensorFloatType (e : TSyntax `tritonExpr) : Option Triton.PointerType :=
  match e with
  | `(tritonExpr| tl.__python_literal($inner:tritonExpr))
  | `(tritonExpr| ($inner:tritonExpr))
  | `(tritonExpr| -$inner:tritonExpr) => literalTensorFloatType inner
  | `(tritonExpr| $n:scientific) =>
      let (mantissa,denominator,exponent) := n.getScientific
      let value := (_root_.Float.ofScientific mantissa denominator exponent).abs
      let lower := _root_.Float.ofBinaryScientific 1 (-126)
      let upper := _root_.Float.ofBinaryScientific 16777215 104
      if value == 0 || value.isInf || (lower ≤ value && value ≤ upper)
      then some .fp32 else some .fp64
  | `(tritonExpr| -inf) => some .fp32
  | `(tritonExpr| float($arg:term))
  | `(tritonExpr| -float($arg:term)) =>
      if arg.raw.isNatLit? == some 0 then some .fp32 else
      match arg with
      | `(term| $s:str) => if ["inf","-inf"].contains s.getString then some .fp32 else none
      | _ => none
  | _ => none

/-- Ordinary integer assignments create tensors. Inline integer literals
remain weak scalars. Match Triton's signed/unsigned 32/64-bit selection before
the value enters register dataflow; never round a large integer through Real. -/
private partial def literalTensorIntegerValue (e : TSyntax `tritonExpr) : Option Int :=
  match e with
  | `(tritonExpr| tl.__python_literal($inner:tritonExpr))
  | `(tritonExpr| ($inner:tritonExpr)) => literalTensorIntegerValue inner
  | `(tritonExpr| -$inner:tritonExpr) => (literalTensorIntegerValue inner).map Int.neg
  | `(tritonExpr| $n:num) => some (Int.ofNat n.getNat)
  | _ => none

private def literalTensorIntegerType (value : Int) : Option Triton.PointerType :=
  if -(2^31) ≤ value && value < 2^31 then some .i32
  else if 2^31 ≤ value && value < 2^32 then some .u32
  else if -(2^63) ≤ value && value < 2^63 then some .i64
  else if 2^63 ≤ value && value < 2^64 then some .u64
  else none

private def checkedIntegerLiteralType (e : TSyntax `tritonExpr) (value : Int) :
    MacroM Triton.PointerType := do
  match literalTensorIntegerType value with
  | some dtype => return dtype
  | none => Macro.throwErrorAt e "integer scalar literal assignment is outside Triton's 64-bit range"

private def assignedValueType (pointers : Types) (values : Values)
    (e : TSyntax `tritonExpr) : MacroM (Option ValueInfo) := do
  if let some value := literalTensorIntegerValue e then
    return some (.typed (← checkedIntegerLiteralType e value))
  match literalTensorFloatType e with
  | some .fp32 => return some (.typed .fp32)
  | some .fp64 =>
      Macro.throwErrorAt e "fp64 scalar literal assignment is not supported by this source frontend"
  | _ => return valueType pointers values e

/-- Preserve the tensor creation performed by an ordinary Python assignment.
The weak scalar used inside an expression must not be materialized this way.
For finite decimal literals, check the intervening Python binary64 rounding
before using the existing exact-literal FP32 node. -/
private partial def materializeLiteral (e : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr) := do
  if let some value := literalTensorIntegerValue e then
    let dtype ← checkedIntegerLiteralType e value
    let dtypeName := mkIdent `dtype
    let magnitude : TSyntax `num := ⟨Syntax.mkNumLit (toString value.natAbs)⟩
    let signed : TSyntax `term ← if value < 0 then `(-($magnitude:num : Int))
      else `(($magnitude:num : Int))
    let signedValue ← `(tritonExpr| $(($signed:term : Int)))
    let unsignedValue ← `(tritonExpr| $(($magnitude:num : Nat)))
    match dtype with
    | .i32 => return ← `(tritonExpr| tl.full([], $signedValue:tritonExpr, $dtypeName:ident=tl.int32))
    | .u32 => return ← `(tritonExpr| tl.full([], $unsignedValue:tritonExpr, $dtypeName:ident=tl.uint32))
    | .i64 => return ← `(tritonExpr| tl.full([], $signedValue:tritonExpr, $dtypeName:ident=tl.int64))
    | .u64 => return ← `(tritonExpr| tl.full([], $unsignedValue:tritonExpr, $dtypeName:ident=tl.uint64))
    | _ => Macro.throwErrorAt e "internal integer literal dtype mismatch"
  unless literalTensorFloatType e == some .fp32 do return e
  let rec infinity (e : TSyntax `tritonExpr) : Bool :=
    match e with
    | `(tritonExpr| tl.__python_literal($inner:tritonExpr))
    | `(tritonExpr| ($inner:tritonExpr))
    | `(tritonExpr| -$inner:tritonExpr) => infinity inner
    | `(tritonExpr| -inf) => Bool.true
    | `(tritonExpr| float($arg:term))
    | `(tritonExpr| -float($arg:term)) =>
        match arg with
        | `(term| $s:str) => ["inf","-inf"].contains s.getString
        | _ => Bool.false
    | _ => Bool.false
  -- Infinity already has an exact source primitive. Positive infinity is a
  -- typed FP32 bit-pattern operation: putting its algebraic placeholder into
  -- tl.full would discard that operation and introduce an undefined register.
  if infinity e then return e
  let rec unwrap (e : TSyntax `tritonExpr) : TSyntax `tritonExpr :=
    match e with
    | `(tritonExpr| tl.__python_literal($inner:tritonExpr)) => unwrap inner
    | _ => e
  match PythonConstants.evaluate (unwrap e) with
  | .error message => Macro.throwErrorAt e message
  | .ok (some value) =>
      let rec negativeSign (e : TSyntax `tritonExpr) : Bool :=
        match e with
        | `(tritonExpr| tl.__python_literal($inner:tritonExpr))
        | `(tritonExpr| ($inner:tritonExpr)) => negativeSign inner
        | `(tritonExpr| -$inner:tritonExpr) => !negativeSign inner
        | _ => Bool.false
      -- A rational literal cannot encode the sign of exact zero. Preserve
      -- Python's -0.0 using the same checked FP32 bit-pattern primitive.
      if value.ideal == 0 && negativeSign e then
        let dtypeName := mkIdent `dtype
        return ← `(tritonExpr| tl.bitcast(tl.full([], 2147483648, $dtypeName:ident=tl.uint32), tl.float32))
      unless PythonConstants.roundBinary 24 (-149) 127 value.ideal ==
          PythonConstants.roundBinary 24 (-149) 127 value.binary64 do
        Macro.throwErrorAt e "scalar literal assignment needs unsupported binary64-to-FP32 double rounding"
  | .ok none => pure ()
  let dtypeName := mkIdent `dtype
  `(tritonExpr| tl.full([], $e:tritonExpr, $dtypeName:ident=tl.float32))

private partial def containsStore (raw : Syntax) : Bool :=
  (raw.isAtom && ["tl.store","tl.store("].contains raw.getAtomVal) || raw.getArgs.any containsStore

/-- Integer memory reads have no floating load event. Identify the actual
source pointer type, rather than trusting a cast's destination type. -/
private def integerLoad (pointers : Types) (raw : Syntax) : Bool := Id.run do
  let e : TSyntax `tritonExpr := ⟨raw⟩
  let source : Option (TSyntax `tritonExpr × Array (TSyntax `tritonMemKwarg)) := match e with
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) => some (p,kwargs)
    | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:tritonDType)) => some (p,kwargs)
    | _ => none
  let some (p,kwargs) := source | return Bool.false
  if kwargs.any (fun kw => dtypeOverride kw.raw) then return Bool.false
  return (pointerType pointers p).any (fun dtype =>
    [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains dtype)

/-- Complete floating load assignments retain their source precision. Nested
floating reads still fail closed; known integer reads can remain in algebraic
expressions. Check their address/mask children too, so a nested floating read
cannot be hidden by an integer outer load or a conversion to integer. -/
private partial def rejectNestedLoads (pointers : Types) (raw : Syntax)
    (completeLoad : Bool := Bool.false) : MacroM Unit := do
  let isLoad := raw.getNumArgs > 0 && raw[0].isAtom && raw[0].getAtomVal == "tl.load("
  if isLoad && !completeLoad && !integerLoad pointers raw then
    Macro.throwErrorAt raw "nested load source precision is unsupported by the current frontend"
  for arg in raw.getArgs do rejectNestedLoads pointers arg

/-- The precise-divide source intrinsic accepts FP32. Unknown family or
FP64 operands cannot be silently relabeled by the fixed-precision result. -/
private partial def checkDivRN (pointers : Types) (values : Values) (raw : Syntax) : MacroM Unit := do
  let e : TSyntax `tritonExpr := ⟨raw⟩
  if let `(tritonExpr| tl.div_rn($a:tritonExpr, $b:tritonExpr)) := e then
    let kinds := [valueType pointers values a, valueType pointers values b]
    unless kinds.any (· == some (.typed .fp32)) && kinds.all (fun kind =>
        kind == some (.typed .fp32) || kind == some .integerLiteral ||
          kind == some .floatingLiteral) do
      Macro.throwErrorAt raw "tl.div_rn requires known FP32 source operands; cast other formats explicitly"
  for arg in raw.getArgs do checkDivRN pointers values arg

private def checkAssignmentLoads (pointers : Types) (values : Values)
    (value : TSyntax `tritonExpr) : MacroM Unit := do
  checkDivRN pointers values value.raw
  -- Negating one complete load keeps its floating load event and then its
  -- ordered negation. Other nested load expressions still fail closed.
  if let `(tritonExpr| -$loaded:tritonExpr) := value then
    let isCompleteLoad := match loaded with
      | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
      | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => Bool.true
      | _ => Bool.false
    if isCompleteLoad then
      rejectNestedLoads pointers loaded.raw Bool.true
      return
  -- The existing integer frontend preserves a direct tl.cast(load(...), dtype)
  -- just like load(...).to(dtype). Integer reads have no floating load event.
  -- Keep arbitrary nested floating loads rejected: their load precision must
  -- not disappear into an algebraic Op during source elaboration.
  if let `(tritonExpr| tl.cast($loaded:tritonExpr, $_:tritonDType)) := value then
    let pointer : Option (TSyntax `tritonExpr) := match loaded with
      | `(tritonExpr| tl.load($p:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
      | `(tritonExpr| tl.load($p:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) => some p
      | _ => none
    if let some p := pointer then
      if (pointerType pointers p).any (fun dt =>
          [.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean].contains dt) then
        rejectNestedLoads pointers loaded.raw Bool.true
        return
  -- Direct load-and-method-cast is a supported complete ComputeExpr as well.
  rejectNestedLoads pointers value.raw Bool.true

/-- A complete load with an explicit method cast is composable in the
existing augmented-arithmetic compiler. Keep arbitrary nested FP loads on the
rejected path so a load precision boundary cannot silently disappear. -/
private def checkAugmentedLoads (pointers : Types) (values : Values)
    (value : TSyntax `tritonExpr) : MacroM Unit := do
  checkDivRN pointers values value.raw
  match value with
  | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*).to($_:tritonDType))
  | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*).to($_:tritonDType)) =>
      checkAssignmentLoads pointers values value
  | _ => rejectNestedLoads pointers value.raw

private partial def rejectPointerRebinding (pointers : Types) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $id:ident = $value:tritonExpr) := raw then
    if let some dtype := lookup pointers id then
      unless pointerType pointers value == some dtype do
        Macro.throwErrorAt raw "pointer alias rebinding inside control flow requires an explicit supported merge"
  if let `(tritonStmt| $id:ident += $value:tritonExpr) := raw then
    if (lookup pointers id).isSome && (pointerType pointers value).isSome then
      Macro.throwErrorAt raw "a pointer update requires an integer offset"
  if let `(tritonStmt| $id:ident -= $value:tritonExpr) := raw then
    if (lookup pointers id).isSome && (pointerType pointers value).isSome then
      Macro.throwErrorAt raw "a pointer update requires an integer offset"
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) := raw then
    for id in #[a,b] ++ more do
      if (lookup pointers id).isSome then
        Macro.throwErrorAt raw "pointer tuple rebinding inside control flow is unsupported"
  for arg in raw.getArgs do rejectPointerRebinding pointers arg

private def convertedValue (pointers : Types) (values : Values) (destination value : TSyntax `tritonExpr)
    (operation : String := "tl.store") : MacroM (TSyntax `tritonExpr) := do
  let some dtype := pointerType pointers destination
    | Macro.throwErrorAt value s!"cannot determine {operation} destination dtype from the call signature"
  if valueType pointers values value == some (.typed dtype) then return value
  let some dt ← dtypeSyntaxOfName? (PointerType.dtypeName dtype)
    | Macro.throwError "unsupported implicit store dtype"
  if dtype == .boolean then
    `(tritonExpr| (($value:tritonExpr).to(tl.int8)) != $((0 : Int)))
  else
    `(tritonExpr| ($value:tritonExpr).to($dt:tritonDType))

private partial def mentions (names : List Name) (raw : Syntax) : Bool :=
  (raw.isIdent && names.contains raw.getId.eraseMacroScopes) || raw.getArgs.any (mentions names)

/-- Conservatively identify writes in a loop, including tuple/reduction
assignments and nested iterators. Unknown productions mentioning a name count
as writes. Only proven immutable entry facts may survive an empty loop. -/
private partial def mayWrite (target : Name) (raw : Syntax) : Bool := Id.run do
  if raw.isIdent || raw.isAtom then return Bool.false
  let args := raw.getArgs
  if args[0]?.any Syntax.isIdent then
    if let some position := args.findIdx? (fun arg => arg.isAtom &&
        ["=",":=","+=","-=","*=","/=","//=","%="].contains arg.getAtomVal.trimAscii.toString) then
      return (args.extract 0 position).any (mentions [target])
  if args.size > 1 && args[0]!.isAtom && args[1]!.isIdent &&
      ["for","tl.for"].contains args[0]!.getAtomVal.trimAscii.toString then
    if args[1]!.getId.eraseMacroScopes == target then return Bool.true
  -- Expressions cannot mutate a scalar binding. Recurse through control-flow
  -- nodes; unsupported statement forms are deliberately not assumed harmless.
  if (raw.getKind.toString.splitOn "tritonStmt").length > 1 then
    match (⟨raw⟩ : TSyntax `tritonStmt) with
    | `(tritonStmt| if $_:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        return (yes ++ no).any (fun s => mayWrite target s.raw)
    | `(tritonStmt| if $_:tritonExpr { $yes:tritonStmt* }) =>
        return yes.any (fun s => mayWrite target s.raw)
    | `(tritonStmt| for $_:ident in range($_:tritonExpr, $_:tritonExpr, $_:tritonExpr) { $body:tritonStmt* })
    | `(tritonStmt| for $_:ident in range($_:tritonExpr, $_:tritonExpr) { $body:tritonStmt* })
    | `(tritonStmt| for $_:ident in range($_:tritonExpr) { $body:tritonStmt* }) =>
        return body.any (fun s => mayWrite target s.raw)
    | `(tritonStmt| tl.atomic_add($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*))
    | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr))
    | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr, $_:tritonMemKwarg $[, $_:tritonMemKwarg]*))
    | `(tritonStmt| tl.store($_:tritonExpr, $_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*)) =>
        return Bool.false
    | _ => return mentions [target] raw
  return args.any (mayWrite target)

private def checkInvariant (invariants : Values) (id : Ident) (value : Option ValueInfo) : MacroM Unit := do
  if let some (_, expected) := invariants.find? (fun p => p.1 == name id) then
    unless value == some expected do
      Macro.throwErrorAt id "loop-carried value requires a stable, known source dtype"

/-- Retain only names with the same known information on every path.
This tracks dtype, never equality of pointer addresses or computed values. -/
private def commonInfo [BEq α] (yes no : List (Name × α)) : List (Name × α) :=
  yes.filter (fun entry => no.any (fun other => other.1 == entry.1 && other.2 == entry.2))

/-- Dtype predicates are compiler metadata. Resolve them only from the current
checked pointer/value provenance; a missing or polymorphic type is not false. -/
private partial def sourceDTypeCondition (pointers : Types) (values : Values)
    (condition : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  let resolve (attr : Ident) (target : TSyntax `tritonDType) (negate : Bool) := do
    let actual := match name attr with
      | .str (.str pointer "dtype") "element_ty"
      | .str (.str pointer "type") "element_ty" => lookup pointers (mkIdent pointer)
      | .str value "dtype" => do
          let (_, .typed kind) ← values.find? (fun entry => entry.1 == value) | none
          pure kind
      | _ => none
    let some actual := actual
      | Macro.throwErrorAt attr "dtype predicate requires a known pointer element or tensor value type"
    let some targetName := dtypeToken target.raw
      | Macro.throwErrorAt target "unsupported dtype predicate target"
    let some expected := sourceKinds.find? (fun kind => PointerType.dtypeName kind == targetName)
      | Macro.throwErrorAt target "unsupported dtype predicate target"
    if [PointerType.inputFloat,PointerType.outputFloat].contains actual ||
        [PointerType.inputFloat,PointerType.outputFloat].contains expected then
      Macro.throwErrorAt condition "dtype predicate requires a specialized source type"
    let result := if negate then actual != expected else actual == expected
    let term ← if result then `(term| Bool.true) else `(term| Bool.false)
    `(tritonExpr| $(($term:term : Bool)))
  match condition with
  | `(tritonExpr| $attr:ident == $target:tritonExpr) =>
      if target.raw.getKind == ``tritonSourceDTypeValue then
        resolve attr ⟨target.raw[0]⟩ Bool.false
      else pure condition
  | `(tritonExpr| $attr:ident != $target:tritonExpr) =>
      if target.raw.getKind == ``tritonSourceDTypeValue then
        resolve attr ⟨target.raw[0]⟩ Bool.true
      else pure condition
  | `(tritonExpr| ($inner:tritonExpr)) =>
      let inner ← sourceDTypeCondition pointers values inner
      `(tritonExpr| ($inner:tritonExpr))
  | _ => pure condition

/-- A supplied Bool term is a constexpr value, not a tensor condition.
Only such conditions may carry different source dtypes into the continuation. -/
private partial def staticBoolCondition (condition : TSyntax `tritonExpr) : Bool :=
  match condition with
  | `(tritonExpr| $(($_:term : Bool))) => Bool.true
  | `(tritonExpr| ($inner:tritonExpr)) => staticBoolCondition inner
  | _ => Bool.false

private def differentKnownFloats (yes no : Values) : Bool :=
  yes.any fun (name,kind) =>
    match kind, no.find? (fun entry => entry.1 == name) with
    | .typed left, some (_, .typed right) =>
        [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5].contains left &&
        [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5].contains right && left != right
    | _, _ => Bool.false

/-- Lower the full block and retain its conservative exit information. A
supported branch joins only identical known dtypes from both exits; one-arm
branches include the unchanged entry environment as their other exit. -/
private partial def lowerWithState (pointers : Types) (values : Values)
    (body : List (TSyntax `tritonStmt)) (invariants : Values := []) :
    MacroM (List (TSyntax `tritonStmt) × Types × Values) := do
  match body with
  | [] => return ([],pointers,values)
  | stmt :: rest =>
      -- Pointer aliases introduced by the source, including aliases used in
      -- loops, are known at this point. Lift eager floating reads using those
      -- actual storage facts, before the precision checker sees arithmetic.
      -- The full suffix and current registers reserve temporary names.
      if let `(tritonStmt| $_:ident = $_:tritonExpr) := stmt then
        let floatingPointers := pointers.filterMap fun (name, kind) =>
          if [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.inputFloat,.outputFloat].contains kind
          then some name else none
        let rec namesIn (raw : Syntax) : List Name :=
          if raw.isIdent then [raw.getId.eraseMacroScopes]
          else raw.getArgs.toList.flatMap namesIn
        let reserved := pointers.map Prod.fst ++ values.map Prod.fst ++
          body.flatMap (fun s => namesIn s.raw)
        let lifted ← NestedLoadAssignments.lower floatingPointers #[stmt] reserved
        if lifted.size > 1 then
          return ← lowerWithState pointers values (lifted.toList ++ rest) invariants
      let mut nextPointers := pointers
      let mut nextValues := values
      let mut stmt := stmt
      let loopBody (iterator : Ident) (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt) × Values) := do
        if (lookup pointers iterator).isSome || values.any (fun p => p.1 == name iterator) then
          Macro.throwErrorAt iterator "loop iterator shadows a tracked pointer or value"
        for st in body do rejectPointerRebinding pointers st.raw
        -- Floating carried values must keep their dtype on every path.
        -- Other facts survive only when no body statement can rebind them;
        -- this includes the zero-trip path and avoids inventing integer SSA
        -- widths for mutated loop registers.
        let floating := values.filter (fun p => match p.2 with
          | .typed dtype => [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.inputFloat,.outputFloat].contains dtype
          | _ => Bool.false)
        let stable := values.filter (fun p =>
          floating.any (fun q => q.1 == p.1) || !body.any (fun st => mayWrite p.1 st.raw))
        let (body,_,_) ← lowerWithState pointers stable body.toList (invariants ++ floating)
        return (body.toArray, stable)
      match stmt with
      | `(tritonStmt| $a:ident, $b:ident = tl.broadcast($x:tritonExpr, $y:tritonExpr))
      | `(tritonStmt| $a:ident, $b:ident := tl.broadcast($x:tritonExpr, $y:tritonExpr)) =>
          -- Broadcasting changes shapes, not element dtypes. Both result
          -- types are read from the entry environment, as in Python tuples.
          if readsTupleTarget (name a) y.raw then
            Macro.throwErrorAt stmt
              "tuple assignment needs unsupported simultaneous register capture"
          checkAssignmentLoads pointers values x
          checkAssignmentLoads pointers values y
          for (id,value) in [(a,x),(b,y)] do
            if (lookup pointers id).isSome then
              Macro.throwErrorAt stmt "pointer tuple assignment needs an explicit supported binding"
            let dtype ← assignedValueType pointers values value
            checkInvariant invariants id dtype
            nextPointers := nextPointers.filter (fun entry => entry.1 != name id)
            nextValues := nextValues.filter (fun entry => entry.1 != name id)
            if let some dtype := dtype then nextValues := (name id,dtype) :: nextValues
      | `(tritonStmt| $a:ident, _ = tl.broadcast($x:tritonExpr, $y:tritonExpr))
      | `(tritonStmt| $a:ident, _ := tl.broadcast($x:tritonExpr, $y:tritonExpr)) =>
          checkAssignmentLoads pointers values x
          checkAssignmentLoads pointers values y
          if (lookup pointers a).isSome then
            Macro.throwErrorAt stmt "pointer tuple assignment needs an explicit supported binding"
          let dtype ← assignedValueType pointers values x
          checkInvariant invariants a dtype
          nextPointers := pointers.filter (fun entry => entry.1 != name a)
          nextValues := values.filter (fun entry => entry.1 != name a)
          if let some dtype := dtype then nextValues := (name a,dtype) :: nextValues
      | `(tritonStmt| $maximum:ident, $position:ident = tl.max($input:tritonExpr $[, $kwargs:tritonReduceKwarg]*))
      | `(tritonStmt| $maximum:ident, $position:ident := tl.max($input:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) =>
          let reduction ← `(tritonExpr| tl.max($input:tritonExpr $[, $kwargs:tritonReduceKwarg]*))
          checkAssignmentLoads pointers values reduction
          -- Track the known FP32 value result separately from the integer
          -- position. The ordinary compiler validates the tuple options and
          -- reduction shape. No index ABI width is inferred here.
          let maximumType := if valueType pointers values input == some (.typed .fp32)
            then some (.typed .fp32) else none
          for (id,dtype) in [(maximum,maximumType),(position,some ValueInfo.integerScalar)] do
            if (lookup pointers id).isSome then
              Macro.throwErrorAt stmt "pointer tuple assignment needs an explicit supported binding"
            checkInvariant invariants id dtype
            nextPointers := nextPointers.filter (fun entry => entry.1 != name id)
            nextValues := nextValues.filter (fun entry => entry.1 != name id)
            if let some dtype := dtype then nextValues := (name id,dtype) :: nextValues
      | `(tritonStmt| $a:ident, $b:ident $[, $ids:ident]* =
          $x:tritonExpr, $y:tritonExpr $[, $rhs:tritonExpr]*) =>
          let targets := #[a,b] ++ ids
          let expressions := #[x,y] ++ rhs
          unless targets.size == expressions.size do
            Macro.throwErrorAt stmt "tuple assignment needs one value per target"
          -- A loop-carried tuple is checked per target below, against the
          -- entry environment. Its dtype must remain invariant, and the
          -- sequential IR still cannot capture an overwritten RHS register.
          if targets.any (fun id => invariants.any (fun entry => entry.1 == name id)) then
            unless expressions.all (fun value => match value with
              | `(tritonExpr| $_:ident) => Bool.true
              | _ => Bool.false) do
              Macro.throwErrorAt stmt "unsupported statement involving a loop-carried source dtype"
          let mut overwritten : List Name := []
          for (id,value) in targets.zip expressions do
            if overwritten.any (fun target => readsTupleTarget target value.raw) then
              Macro.throwErrorAt stmt
                "tuple assignment needs unsupported simultaneous register capture"
            overwritten := name id :: overwritten
          -- Python evaluates all RHSs in the entry environment. Preserve
          -- unrelated ABI facts, and replace only the explicitly bound names.
          for (id,value) in targets.zip expressions do
            if (lookup pointers id).isSome || (pointerType pointers value).isSome then
              Macro.throwErrorAt stmt "pointer tuple assignment needs an explicit supported binding"
            checkAssignmentLoads pointers values value
            let dtype ← assignedValueType pointers values value
            checkInvariant invariants id dtype
            nextPointers := nextPointers.filter (fun entry => entry.1 != name id)
            nextValues := nextValues.filter (fun entry => entry.1 != name id)
            if let some dtype := dtype then
              nextValues := (name id, dtype) :: nextValues
          let materialized ← expressions.mapM materializeLiteral
          let x := materialized[0]!
          let y := materialized[1]!
          let rhs := materialized.extract 2 materialized.size
          stmt ← `(tritonStmt| $a:ident, $b:ident $[, $ids:ident]* =
            $x:tritonExpr, $y:tritonExpr $[, $rhs:tritonExpr]*)
      -- The grammar has a dedicated positional-axis assignment production.
      -- Its reduction dtype must enter the same dataflow as a generic RHS;
      -- otherwise a later store spuriously adds another destination cast.
      | `(tritonStmt| $id:ident = tl.max($value:tritonExpr, $axisIndex:num))
      | `(tritonStmt| $id:ident := tl.max($value:tritonExpr, $axisIndex:num)) =>
          let reduction ← `(tritonExpr| tl.max($value:tritonExpr, axis = $axisIndex:num))
          checkAssignmentLoads pointers values reduction
          let dtype := valueType pointers values reduction
          checkInvariant invariants id dtype
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := dtype then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident = $value:tritonExpr) =>
          checkAssignmentLoads pointers values value
          let dtype ← assignedValueType pointers values value
          let materialized ← materializeLiteral value
          stmt ← `(tritonStmt| $id:ident = $materialized:tritonExpr)
          checkInvariant invariants id dtype
          nextPointers := pointers.filter (fun entry => entry.1 != name id)
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some dtype := pointerType pointers value then
            nextPointers := (name id, dtype) :: nextPointers
          if let some dtype := dtype then
            nextValues := (name id, dtype) :: nextValues
      | `(tritonStmt| $id:ident += $value:tritonExpr) =>
          checkAugmentedLoads pointers values value
          let expression ← `(tritonExpr| $id:ident + $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident -= $value:tritonExpr) =>
          checkAugmentedLoads pointers values value
          let expression ← `(tritonExpr| $id:ident - $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident *= $value:tritonExpr) =>
          checkAugmentedLoads pointers values value
          let expression ← `(tritonExpr| $id:ident * $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| $id:ident /= $value:tritonExpr) =>
          checkAugmentedLoads pointers values value
          let expression ← `(tritonExpr| $id:ident / $value:tritonExpr)
          let dtype := valueType pointers values expression
          checkInvariant invariants id dtype
          nextValues := values.filter (fun entry => entry.1 != name id)
          if let some kind := dtype then nextValues := (name id,kind) :: nextValues
          if (lookup pointers id).isNone then
            nextPointers := pointers.filter (fun entry => entry.1 != name id)
      | `(tritonStmt| tl.debug_barrier()) =>
          -- A barrier does not assign source registers or change their dtypes.
          -- Preserve the checked values across the scheduling operation.
          pure ()
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
          checkDivRN pointers values stmt.raw
          rejectNestedLoads pointers value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
          checkDivRN pointers values stmt.raw
          rejectNestedLoads pointers value.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
          -- A positional mask (including None) has its own parser production.
          -- Apply the same destination conversion and preserve register types.
          checkDivRN pointers values stmt.raw
          rejectNestedLoads pointers value.raw
          rejectNestedLoads pointers mask.raw
          let value ← convertedValue pointers values destination value
          stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| tl.atomic_add($destination:tritonExpr, $value:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
          checkDivRN pointers values stmt.raw
          rejectNestedLoads pointers value.raw
          -- Triton converts the atomic operand to the pointed-to storage type
          -- before selecting its atomic addition primitive (semantic.py,
          -- atom_red_typechecking_impl). Keep that conversion in the program.
          let value ← convertedValue pointers values destination value "tl.atomic_add"
          stmt ← `(tritonStmt| tl.atomic_add($destination:tritonExpr, $value:tritonExpr $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          let condition ← sourceDTypeCondition pointers values condition
          checkDivRN pointers values condition.raw
          rejectNestedLoads pointers condition.raw
          for st in yes ++ no do rejectPointerRebinding pointers st.raw
          let (yes,yesPointers,yesValues) ← lowerWithState pointers values yes.toList invariants
          let (no,noPointers,noValues) ← lowerWithState pointers values no.toList invariants
          if staticBoolCondition condition && differentKnownFloats yesValues noValues then
            -- A constexpr branch selects a source type. Lower the continuation
            -- in that path's actual type environment before merging exits.
            -- This never permits a tensor condition to merge different dtypes.
            let (yesRest,yesFinalPointers,yesFinalValues) ←
              lowerWithState yesPointers yesValues rest invariants
            let (noRest,noFinalPointers,noFinalValues) ←
              lowerWithState noPointers noValues rest invariants
            let yes := (yes ++ yesRest).toArray
            let no := (no ++ noRest).toArray
            let fork ← `(tritonStmt| if $condition { $yes* } else { $no* })
            return ([fork],commonInfo yesFinalPointers noFinalPointers,
              commonInfo yesFinalValues noFinalValues)
          let yes := yes.toArray
          let no := no.toArray
          stmt ← `(tritonStmt| if $condition { $yes* } else { $no* })
          nextPointers := commonInfo yesPointers noPointers
          nextValues := commonInfo yesValues noValues
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          let condition ← sourceDTypeCondition pointers values condition
          checkDivRN pointers values condition.raw
          rejectNestedLoads pointers condition.raw
          for st in yes do rejectPointerRebinding pointers st.raw
          let (yes,yesPointers,yesValues) ← lowerWithState pointers values yes.toList invariants
          if staticBoolCondition condition && differentKnownFloats yesValues values then
            let (yesRest,yesFinalPointers,yesFinalValues) ←
              lowerWithState yesPointers yesValues rest invariants
            let (noRest,noFinalPointers,noFinalValues) ←
              lowerWithState pointers values rest invariants
            let yes := (yes ++ yesRest).toArray
            let no := noRest.toArray
            let fork ← `(tritonStmt| if $condition { $yes* } else { $no* })
            return ([fork],commonInfo yesFinalPointers noFinalPointers,
              commonInfo yesFinalValues noFinalValues)
          let yes := yes.toArray
          stmt ← `(tritonStmt| if $condition { $yes* })
          nextPointers := commonInfo yesPointers pointers
          nextValues := commonInfo yesValues values
      | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
          checkDivRN pointers values start.raw
          checkDivRN pointers values stop.raw
          checkDivRN pointers values step.raw
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($start, $stop, $step) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $body:tritonStmt* }) =>
          checkDivRN pointers values start.raw
          checkDivRN pointers values stop.raw
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($start, $stop) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $body:tritonStmt* }) =>
          checkDivRN pointers values stop.raw
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in range($stop) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in tl.static_range($($n:term)) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in tl.static_range($($n)) { $body* })
          nextValues := stable
      | `(tritonStmt| for $i:ident in tl.static_range($n:num) { $body:tritonStmt* }) =>
          let (body, stable) ← loopBody i body
          stmt ← `(tritonStmt| for $i in tl.static_range($n:num) { $body* })
          nextValues := stable
      | _ =>
          if mentions (invariants.map Prod.fst) stmt.raw then
            Macro.throwErrorAt stmt "unsupported statement involving a loop-carried source dtype"
          rejectNestedLoads pointers stmt.raw
          rejectPointerRebinding pointers stmt.raw
          if containsStore stmt.raw then
            Macro.throwErrorAt stmt "unsupported store control flow in call signature"
          -- An unsupported statement can assign a tracked value through a
          -- form not understood above. Forget value types conservatively.
          nextValues := []
      let (rest,finalPointers,finalValues) ← lowerWithState nextPointers nextValues rest invariants
      return (stmt :: rest,finalPointers,finalValues)

/-- Every store resolves its destination dtype. Supported control flow keeps
only dtype facts proved by both exits; unsupported forms forget value facts. -/
def lower (pointers : Types) (values : Values)
    (body : List (TSyntax `tritonStmt)) (invariants : Values := []) : MacroM (List (TSyntax `tritonStmt)) := do
  return (← lowerWithState pointers values body invariants).1

end VeriTile.Triton.DSL.TypedStores
