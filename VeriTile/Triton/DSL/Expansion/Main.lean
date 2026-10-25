/-
VeriTile.Triton.DSL.Expansion

A `triton { ... }` macro that embeds Triton-style kernel syntax inside Lean,
modelled on the `arm64 { ... }` style used in arm-in-lean.

Example:

  def naiveSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel := triton {
    pid  := tl.program_id(0)
    offs := pid * $(N) + tl.arange($(N))
    x    := tl.load($(xReg) + offs)
    e    := tl.exp(x)
    s    := tl.sum(e)
    y    := e / s
    tl.store($(yReg) + offs, y)
  }

Conventions:
  * Bare identifiers in expression position → register references
    (`pid`, `x`, `e`, etc. become `Op.ref "pid"`, `Op.ref "x"`, ...).
  * Memory accesses use pointer-like kernel parameters:
    `$(xReg) + offs` builds a pointer tile in pointer context, and
    `tl.load(ptr)` / `tl.store(ptr, value)` consume pointer-valued expressions.
  * `$(<lean-term>)` is context-sensitive: address/shape/index contexts lower it
    to `.nat`, while data/scalar contexts lower it to `.real`. Ambiguous uses
    should be rejected instead of guessed.
  * Numeric literals become `Op.const`.
  * Integer `//` and `%` inside the block model runtime tensor operations.
    Signed division truncates toward zero. Python constexpr calculations must
    be evaluated inside a Lean antiquotation; use `Int.fdiv` / `Int.fmod` for
    constexpr signed division/remainder. See Core.Integer for this boundary.
  * Statements are separated by newlines (no explicit terminator).

Currently supported expressions: `tl.program_id(_)`, `tl.arange(_)` /
`tl.arange(start, end)`, `tl.exp(_)`, `tl.exp2(_)`, `tl.log(_)`, `tl.log2(_)`,
`tl.sigmoid(_)`, `tl.sqrt(_)`, `tl.math.rsqrt(_)`/`tl.rsqrt(_)`, `tl.tanh(_)`, `tl.sin(_)`, `tl.cos(_)`,
`tl.tan(_)`, `tl.atan(_)`, `tl.cosh(_)`, `tl.sinh(_)`, `tl.erf(_)`,
`tl.extra.cuda.libdevice.erf(_)`, `tl.max(_)`, `tl.sum(_)`, `tl.load(ptrExpr)`,
binary `+ - * /`, parens, identifiers, numerals, antiquotation (`$(t)` for
Lean values in typed contexts / `RegionName` in pointer context).
Pointer arithmetic supports `ptr + nat` and `nat + ptr`.

The two-argument `tl.arange(start, end)` lowers to `start + tl.arange(end - start)`
at macro time (no new AST constructor). The literal-0 special case
`tl.arange(0, e)` collapses to `tl.arange(e)`, producing an AST identical to the
single-argument form so existing proofs (e.g. via `scatter_readback`) remain
applicable verbatim.

Currently supported statements: assignment (`name := expr`),
`tl.store(ptrExpr, value)`.

`Kernel.inputs` / `Kernel.outputs` are auto-populated by scanning the body for
`tl.load(...)` (input regions) and `tl.store(...)` (output regions). Order
follows body occurrence; no macro-time dedup since regions are Lean terms
(possibly equal at runtime but not statically). `Kernel.inputs/outputs` is
metadata only (not consumed by `exec`), so duplicates are harmless.
An optional top-level `tl.ports inputs=[...], outputs=[...]` declares the
ordered interface explicitly, including optional regions. It emits no runtime
statement, changes no memory operation, and does not establish memory safety.

This module owns macro expansion only. Surface syntax, macro-time typing
helpers, and metadata scanning live in sibling `DSL.*` modules.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Core.Integer
import VeriTile.Triton.Core.FixedInteger
import VeriTile.Triton.Core.SignedCast
import VeriTile.Triton.DSL.Metadata
import VeriTile.Triton.DSL.Inference
import VeriTile.Triton.DSL.RandomProfile
import VeriTile.Triton.DSL.FiniteHelperSource
import VeriTile.Triton.DSL.WelfordSource
import VeriTile.Triton.DSL.Expansion.Common
import VeriTile.Triton.DSL.Expansion.Memory
import VeriTile.Triton.DSL.Expansion.Compute
import VeriTile.Triton.DSL.Expansion.DType
import VeriTile.Triton.DSL.Expansion.Control
import VeriTile.Triton.DSL.Expansion.InlineAsm
import VeriTile.Triton.DSL.Expansion.DirectRandom
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.Typing

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

set_option maxHeartbeats 400000

namespace VeriTile.Triton.DSL

/-! ## Expansion -/

def extractDTypeKwarg (kwargs : TSyntaxArray `tritonMemKwarg) :
    MacroM (Option DInfo) := do
  for kw in kwargs do
    match kw with
    | `(tritonMemKwarg| $name:ident = $dt:tritonDType) =>
        if name.getId.toString == "dtype" then
          return some (← expandDType dt)
    | _ => pure ()
  return none

def parseMaxReturnIndicesKwargs (dims : List (TSyntax `term))
    (kwargs : TSyntaxArray `tritonReduceKwarg) : MacroM Nat := do
  let mut seenAxis : Bool := Bool.false
  let mut seenReturn : Bool := Bool.false
  let mut axisIdx : Nat := 0
  for kw in kwargs do
    match kw with
    | `(tritonReduceKwarg| $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max return_indices: duplicate axis"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError "tl.max return_indices: axis out of bounds"
        axisIdx := n.getNat
    | `(tritonReduceKwarg| axis = $n:num) =>
        if seenAxis then
          Macro.throwError "tl.max return_indices: duplicate axis"
        seenAxis := Bool.true
        if n.getNat ≥ dims.length then
          Macro.throwError "tl.max return_indices: axis out of bounds"
        axisIdx := n.getNat
    | `(tritonReduceKwarg| return_indices=True) =>
        seenReturn := Bool.true
    | `(tritonReduceKwarg| return_indices=False) =>
        Macro.throwError "tl.max tuple binding requires `return_indices=True`"
    | `(tritonReduceKwarg| keep_dims = false) =>
        Macro.throwError "tl.max return_indices: `keep_dims` is not supported"
    | `(tritonReduceKwarg| keep_dims = true) =>
        Macro.throwError "tl.max return_indices: `keep_dims` is not supported"
    | `(tritonReduceKwarg| $name:ident = $value:tritonExpr) =>
        let nm := name.getId.toString
        if nm == "return_indices_tie_break_left" then
          match value with
          | `(tritonExpr| True) => pure ()
          | _ => Macro.throwError "tl.max return_indices: only literal True leftmost tie breaking is supported"
        else
          Macro.throwError
            ("tl.max return_indices: unsupported kwarg `" ++ nm ++ "`")
    | _ => Macro.throwUnsupported
  unless seenReturn do
    Macro.throwError "tl.max tuple binding requires `return_indices=True`"
  unless seenAxis do
    Macro.throwError "tl.max return_indices requires an explicit axis"
  pure axisIdx

private def typedRegionAntiquote? (r : TSyntax `term) :
    MacroM (Option (TSyntax `term × DInfo)) := do
  let dtype? ←
    match r with
    | `(($_:term : Region .real)) => pure (some DInfo.real)
    | `(($_:term : Region .fp32)) => pure (some DInfo.fp32)
    | `(($_:term : Region .fp16)) => pure (some DInfo.fp16)
    | `(($_:term : Region .bf16)) => pure (some DInfo.bf16)
    | `(($_:term : Region .f8e4)) => pure (some DInfo.f8e4)
    | `(($_:term : Region .f8e5)) => pure (some DInfo.f8e5)
    | `(($_:term : Region .int)) => pure (some DInfo.int)
    | `(($_:term : Region .nat)) => pure (some DInfo.nat)
    | `(($_:term : Region .bool)) => pure (some DInfo.bool)
    | `(($_:term : Region TileDType.real)) => pure (some DInfo.real)
    | `(($_:term : Region TileDType.fp32)) => pure (some DInfo.fp32)
    | `(($_:term : Region TileDType.fp16)) => pure (some DInfo.fp16)
    | `(($_:term : Region TileDType.bf16)) => pure (some DInfo.bf16)
    | `(($_:term : Region TileDType.f8e4)) => pure (some DInfo.f8e4)
    | `(($_:term : Region TileDType.f8e5)) => pure (some DInfo.f8e5)
    | `(($_:term : Region TileDType.int)) => pure (some DInfo.int)
    | `(($_:term : Region TileDType.nat)) => pure (some DInfo.nat)
    | `(($_:term : Region TileDType.bool)) => pure (some DInfo.bool)
    | _ => pure none
  let some dtype := dtype?
    | pure none
  pure (some (r, dtype))

private def expandIdentRefByName (env : Env) (name : String) : MacroM EOut := do
  let (dtype, shape) ← lookupEnv env name
  let s : TSyntax `term := Syntax.mkStrLit name
  let dt ← dtype.term
  let sh ← shape.term
  let term ← `(Op.ref $dt $sh $s)
  if dtype == .fp32Argument then
    let bridge ← `(Op.castFloat FloatDType.fp32 FloatDType.real $term)
    return ⟨bridge, .real, shape, some (← fp32ComputeExpr bridge), some .fp32⟩
  match lookupComputeDType? env name with
  | some .fp32 =>
      pure ⟨term, dtype, shape, some (← fp32ComputeExpr term), some .fp32⟩
  | some .real =>
      pure ⟨term, dtype, shape, some (← fp64ComputeExpr term), some .real⟩
  | some _ =>
      pure ⟨term, dtype, shape,
        some (← `(ComputeExpr.opaque ("runtime compute value `" ++ $s ++ "`"))),
        lookupComputeDType? env name⟩
  | none =>
      pure ⟨term, dtype, shape, none, none⟩

private partial def nonnegativeIntTermAsNat? (term : TSyntax `term) :
    MacroM (Option (TSyntax `term)) := do
  match term with
  | `(Op.castNatToInt $e) =>
      pure (some e)
  | `(Op.constInt (Int.ofNat $e)) =>
      pure (some (← `(Op.constNat $e)))
  | `(Op.constInt $n:num) =>
      pure (some (← `(Op.constNat $n)))
  | `(Op.add NumericDType.int $bc $a $b) => do
      match ← nonnegativeIntTermAsNat? a, ← nonnegativeIntTermAsNat? b with
      | some aNat, some bNat =>
          pure (some (← `(Op.add NumericDType.nat $bc $aNat $bNat)))
      | _, _ => pure none
  | `(Op.mul NumericDType.int $bc $a $b) => do
      match ← nonnegativeIntTermAsNat? a, ← nonnegativeIntTermAsNat? b with
      | some aNat, some bNat =>
          pure (some (← `(Op.mul NumericDType.nat $bc $aNat $bNat)))
      | _, _ => pure none
  | _ =>
      pure none

private def pointerOffsetIntAsNat (out : EOut) : MacroM EOut := do
  match ← nonnegativeIntTermAsNat? out.term with
  | some natTerm => pure { out with term := natTerm, dtype := .nat }
  | none => pure { out with term := ← `(Op.castIntToNat $out.term), dtype := .nat }

private def unsignedCastWidth? : TSyntax `tritonDType → Option Nat
  | `(tritonDType| tl.uint8) => some 8
  | `(tritonDType| tl.uint16) => some 16
  | `(tritonDType| tl.uint32) => some 32
  | `(tritonDType| tl.uint64) => some 64
  | _ => none

private def signedCastWidth? : TSyntax `tritonDType → Option Nat
  | `(tritonDType| tl.int8) => some 8
  | `(tritonDType| tl.int16) => some 16
  | `(tritonDType| tl.int32) => some 32
  | `(tritonDType| tl.int64) => some 64
  | _ => none

/-- Numeric float-to-integer casts keep their own source precision because
integer results have no enclosing floating compute channel. -/
private def floatingIntegerCast? (dst : TSyntax `tritonDType) (src : EOut) :
    MacroM (Option EOut) := do
  match src.dtype with
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ => pure ()
  | _ => return none
  let signedWidth := signedCastWidth? dst
  let some width := signedWidth.orElse (fun _ => unsignedCastWidth? dst) | return none
  let bits : TSyntax `term := ⟨Syntax.mkNumLit (toString width)⟩
  let sourceDType ← src.dtype.floatProof
  let precision ← match src.computeDType? with
    | some .fp32 => `(some IndexComputePrecision.fp32)
    | some .real => `(some IndexComputePrecision.fp64)
    | _ => `(none)
  let child ← if src.computeDType?.isSome then src.scopedFloatingTerm else src.floatingTerm
  if signedWidth.isSome then
    let dtype := if width == 8 then DInfo.int8 else if width == 32 then DInfo.int32 else DInfo.int
    return some ⟨← `(Op.floatToInt $precision $sourceDType $bits $child),
      dtype, src.shape, none, none⟩
  return some ⟨← `(Op.floatToNat $precision $sourceDType $bits $child),
    .nat, src.shape, none, none⟩

/-- Machine casts must be evaluated before forgetting source storage width.
Signed i8/i16/i32 narrowing keeps the low word and reinterprets its sign.
This preserves explicit casts; it does not attach i16 width to all arithmetic. -/
private def machineIntegerCast? (dst : TSyntax `tritonDType) (src : EOut) :
    MacroM (Option EOut) := do
  if let some cast ← floatingIntegerCast? dst src then return some cast
  if let `(tritonDType| tl.int8) := dst then
    match src.dtype with
    | .nat => return some ⟨← `(Op.intToSigned 8 (Op.castNatToInt $src.term)), .int8,
        src.shape, none, none⟩
    | .int | .int32 => return some ⟨← `(Op.intToSigned 8 $src.term), .int8,
        src.shape, none, none⟩
    | .bool => return some ⟨← `(Op.boolToInt $src.term), .int8,
        src.shape, none, none⟩
    | .int8 => return some src
    | _ => return none
  if let `(tritonDType| tl.int16) := dst then
    let term ← match src.dtype with
      | .nat => `(Op.intToSigned 16 (Op.castNatToInt $src.term))
      | .int | .int8 | .int32 => `(Op.intToSigned 16 $src.term)
      | .bool => `(Op.intToSigned 16 (Op.boolToInt $src.term))
      | _ => return none
    return some ⟨term, .int, src.shape, none, none⟩
  if let `(tritonDType| tl.int32) := dst then
    match src.dtype with
    | .nat => return some ⟨← `(Op.intToSigned 32 (Op.castNatToInt $src.term)), .int32,
        src.shape, none, none⟩
    | .int => return some ⟨← `(Op.intToSigned 32 $src.term), .int32,
        src.shape, none, none⟩
    | .bool => return some ⟨← `(Op.boolToInt $src.term), .int32,
        src.shape, none, none⟩
    | .int32 => return some src
    | .int8 => return some { src with dtype := .int32 }
    | _ => return none
  let some width := unsignedCastWidth? dst | return none
  let bits : TSyntax `term := ⟨Syntax.mkNumLit (toString width)⟩
  let term ← match src.dtype with
    | .int | .int8 | .int32 => `(Op.intToUnsigned $bits $src.term)
    | .nat => `(Op.natToUnsigned $bits $src.term)
    | _ => return none
  return some ⟨term, .nat, src.shape, none, none⟩

private def signedMachineCast (dt : TSyntax `tritonDType) : Bool :=
  match dt with
  | `(tritonDType| tl.int8) | `(tritonDType| tl.int16) | `(tritonDType| tl.int32) => Bool.true
  | _ => Bool.false

/-- Facts about immutable, external constexpr parameters on the current path.
The optional string identifies a string-equality atom; `none` is a Bool atom.
Register bindings shadow these parameters, so facts are only consulted while
the corresponding name is absent from the register environment. -/
abbrev StaticBoolFacts := List ((Name × Option String) × Bool)


mutual

partial def expandNatExpectedExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM EOut := do
  if stx.raw.getKind == ``tritonDottedIdentMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 4 then
      let i : TSyntax `ident := ⟨args[0]⟩
      match dottedToBaseName? i with
      | some name =>
          let dt : TSyntax `tritonDType := ⟨args[2]⟩
          let e' ← if (unsignedCastWidth? dt).isSome then expandExpr env stx
            else expandIdentRefByName env name
          ensureDType .nat e'.dtype "nat expression"
          pure e'
      | none => Macro.throwUnsupported
    else
      Macro.throwUnsupported
  else
  if stx.raw.getKind == ``tritonMethodCastElementTy ||
      stx.raw.getKind == ``tritonMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdentSpaced ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdentSpaced then
    let args := stx.raw.getArgs
    if h : 0 < args.size then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      expandNatExpectedExpr env e
    else
      Macro.throwUnsupported
  else
  match methodCast? stx with
  | some (e, dt) =>
      let dst ← expandDType dt
      if dst == .int then
        expandNatExpectedExpr env e
      else
        let e' ← expandExpr env stx
        ensureDType .nat e'.dtype "nat expression"
        pure e'
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| $i:ident) =>
      let rawName := i.getId.toString
      let userName := i.getId.getString!
      let erasedName := i.getId.eraseMacroScopes.toString
      if env.any (fun entry => entry.1 == rawName || entry.1 == userName || entry.1 == erasedName) then
        let name :=
          if env.any (fun entry => entry.1 == rawName) then rawName
          else if env.any (fun entry => entry.1 == userName) then userName
          else erasedName
        let (dtype, shape) ← lookupEnv env name
        ensureDType .nat dtype "nat expression"
        let s : TSyntax `term := Syntax.mkStrLit name
        let sh ← shape.term
        pure ⟨← `(Op.ref TileDType.nat $sh $s), .nat, shape, none, none⟩
      else
        let t : TSyntax `term := ⟨i.raw⟩
        pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| $($t:term)) =>
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandNatExpectedExpr env e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.add NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.sub NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.mul NumericDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.floorDiv IntegralDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| _div_up($a:tritonExpr, $b:tritonExpr)) => do
      let one ← `(Op.constNat 1)
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      let num ← `(Op.sub NumericDType.nat $bc
        (Op.add NumericDType.nat $bc $a'.term $b'.term) $one)
      pure ⟨← `(Op.floorDiv IntegralDType.nat $bc $num $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "nat expression"
      pure ⟨← `(Op.mod IntegralDType.nat $bc $a'.term $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat) (some mask)
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      let dst ← expandDType dt
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some dst) (some mask)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_dt:ident)) => do
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat) (some mask)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_base:ident.$_field:ident)) => do
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat) (some mask)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat)
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      let dst ← expandDType dt
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some dst)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_dt:ident)) => do
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_base:ident.$_field:ident)) => do
      let e' ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .nat)
      ensureDType .nat e'.dtype "nat expression"
      pure e'
  | _ => do
      let e' ← expandExpr env stx
      ensureDType .nat e'.dtype "nat expression"
      pure e'

partial def expandIntExpectedExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM EOut := do
  let natToInt (e' : EOut) : MacroM EOut := do
    match e'.dtype with
    | .int | .int8 | .int32 => pure e'.forgetStorageWidth
    | .nat =>
        pure ⟨← `(Op.castNatToInt $e'.term), .int, e'.shape,
          e'.computeTerm, e'.computeDType?⟩
    | _ =>
        ensureDType .int e'.dtype "int expression"
        pure e'
  if stx.raw.getKind == ``tritonDottedIdentMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 4 then
      let i : TSyntax `ident := ⟨args[0]⟩
      match dottedToBaseName? i with
      | some name =>
          let dt : TSyntax `tritonDType := ⟨args[2]⟩
          if (unsignedCastWidth? dt).isSome || signedMachineCast dt then natToInt (← expandExpr env stx)
          else natToInt (← expandIdentRefByName env name)
      | none => Macro.throwUnsupported
    else
      Macro.throwUnsupported
  else
  if stx.raw.getKind == ``tritonMethodCastElementTy ||
      stx.raw.getKind == ``tritonMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdentSpaced ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdentSpaced then
    let args := stx.raw.getArgs
    if h : 0 < args.size then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      if methodCastTargetName? stx == some `tl.int32 then
        natToInt (← expandExpr env stx)
      else expandIntExpectedExpr env e
    else
      Macro.throwUnsupported
  else
  match methodCast? stx with
  | some (e, dt) =>
      let dst ← expandDType dt
      if dst == .nat then
        natToInt (← expandExpr env stx)
      else
        natToInt (← expandExpr env stx)
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      pure ⟨← `(Op.constInt $n), .int, SInfo.scalar, none, none⟩
  | `(tritonExpr| $i:ident) =>
      let rawName := i.getId.toString
      let userName := i.getId.getString!
      let erasedName := i.getId.eraseMacroScopes.toString
      if env.any (fun entry => entry.1 == rawName || entry.1 == userName || entry.1 == erasedName) then
        let name :=
          if env.any (fun entry => entry.1 == rawName) then rawName
          else if env.any (fun entry => entry.1 == userName) then userName
          else erasedName
        natToInt (← expandIdentRefByName env name)
      else
        let t : TSyntax `term := ⟨i.raw⟩
        pure ⟨← `(Op.constInt (Int.ofNat $t)), .int, SInfo.scalar, none, none⟩
  | `(tritonExpr| $(($t:term : Int))) =>
      pure ⟨← `(Op.constInt $t), .int, SInfo.scalar, none, none⟩
  | `(tritonExpr| $($t:term)) =>
      pure ⟨← `(Op.constInt (Int.ofNat $t)), .int, SInfo.scalar, none, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandIntExpectedExpr env e
  | `(tritonExpr| -$e:tritonExpr) => do
      let e' ← expandIntExpectedExpr env e
      let (bc, shape) ← broadcastTerm SInfo.scalar e'.shape "int negation"
      pure ⟨← `(Op.sub NumericDType.int $bc (Op.constInt 0) $e'.term),
        .int, shape, none, none⟩
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a' ← expandIntExpectedExpr env a
      let b' ← expandIntExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "int expression"
      pure ⟨← `(Op.add NumericDType.int $bc $a'.term $b'.term),
        .int, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      let a' ← expandIntExpectedExpr env a
      let b' ← expandIntExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "int expression"
      pure ⟨← `(Op.sub NumericDType.int $bc $a'.term $b'.term),
        .int, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      let a' ← expandIntExpectedExpr env a
      let b' ← expandIntExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "int expression"
      pure ⟨← `(Op.mul NumericDType.int $bc $a'.term $b'.term),
        .int, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      let a' ← expandIntExpectedExpr env a
      let b' ← expandIntExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "int expression"
      pure ⟨← `(Op.truncDiv $bc $a'.term $b'.term),
        .int, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      let a' ← expandIntExpectedExpr env a
      let b' ← expandIntExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "int expression"
      pure ⟨← `(Op.truncMod $bc $a'.term $b'.term),
        .int, outShape, none, none⟩
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .int) (some mask)
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some .int)
  | _ => do
      natToInt (← expandExpr env stx)

/-- The rand source profile determines an inline integer load's carrier.
Unsigned loads use Nat and then inject into Int; other operands retain the
existing exact mathematical integer expressions and checked source domain. -/
partial def expandRandIntegerExpr (env : Env) (sourceType : Random.IntegerType)
    (stx : TSyntax `tritonExpr) : MacroM EOut := do
  let isUnsigned := sourceType == .u32 || sourceType == .u64
  let loadedToInt (e : EOut) : MacroM EOut := do
    if isUnsigned then
      ensureDType .nat e.dtype "tl.rand unsigned load"
      return ⟨← `(Op.castNatToInt $e.term), .int, e.shape, none, none⟩
    let e := e.forgetStorageWidth
    ensureDType .int e.dtype "tl.rand signed load"
    return e
  match stx with
  | `(tritonExpr| ($e:tritonExpr)) => expandRandIntegerExpr env sourceType e
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      loadedToInt (← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some (if isUnsigned then .nat else .int)))
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      loadedToInt (← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some (if isUnsigned then .nat else .int)) (some mask))
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) =>
      let a' ← expandRandIntegerExpr env sourceType a
      let b' ← expandRandIntegerExpr env sourceType b
      let (bc, shape) ← broadcastTerm a'.shape b'.shape "tl.rand integer operand"
      return ⟨← `(Op.add NumericDType.int $bc $a'.term $b'.term), .int, shape, none, none⟩
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let a' ← expandRandIntegerExpr env sourceType a
      let b' ← expandRandIntegerExpr env sourceType b
      let (bc, shape) ← broadcastTerm a'.shape b'.shape "tl.rand integer operand"
      return ⟨← `(Op.mul NumericDType.int $bc $a'.term $b'.term), .int, shape, none, none⟩
  | _ => expandIntExpectedExpr env stx

private partial def valueTypedAntiquote? (stx : TSyntax `tritonExpr) : MacroM Bool := do
  match stx with
  | `(tritonExpr| $(($_:term : Nat))) => pure Bool.true
  | `(tritonExpr| $(($_:term : Int))) => pure Bool.true
  | `(tritonExpr| $(($_:term : Bool))) => pure Bool.true
  | `(tritonExpr| $(($_:term : ℝ))) => pure Bool.true
  | `(tritonExpr| $(($_:term : Real))) => pure Bool.true
  | _ => pure Bool.false

private partial def sameRaw (a b : Syntax) : Bool :=
  toString a == toString b

private partial def expandArangeRangeWithEnv (env : Env)
    (s e : TSyntax `tritonExpr) : MacroM EOut := do
  match s, e with
  | `(tritonExpr| $i:ident * $($step:term)),
    `(tritonExpr| ($j:ident + $(1)) * $($step':term))
  | `(tritonExpr| $i:ident * $($step:term)),
    `(tritonExpr| ($j:ident + 1) * $($step':term)) =>
      if i.getId.eraseMacroScopes != j.getId.eraseMacroScopes ||
          !sameRaw step.raw step'.raw then
        expandArangeRange s e
      else
        let s' ← expandNatExpectedExpr env s
        let stepTerm ← `(($step : Nat))
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              $s'.term (Op.arange $stepTerm)),
          .nat, SInfo.vec stepTerm, none, none⟩
  | _, _ => expandArangeRange s e

partial def expandStaticPtrExpr (env : Env) (stx : TSyntax `tritonExpr) :
    MacroM (Option StaticPtrOut) := do
  if ← valueTypedAntiquote? stx then
    return none
  match stx with
  | `(tritonExpr| $($r:term)) =>
      let zero ← `(Op.constNat 0)
      match ← typedRegionAntiquote? r with
      | some (region, dtype) =>
          pure (some ⟨region, some dtype, zero, SInfo.scalar, Bool.true⟩)
      | none =>
          pure (some ⟨r, none, zero, SInfo.scalar, Bool.true⟩)
  | `(tritonExpr| $r:ident) =>
      let rawName := r.getId.toString
      let userName := r.getId.getString!
      let erasedName := r.getId.eraseMacroScopes.toString
      if env.any (fun entry => entry.1 == rawName || entry.1 == userName || entry.1 == erasedName) then
        pure (none : Option StaticPtrOut)
      else
        let zero ← `(Op.constNat 0)
        let region : TSyntax `term := ⟨r.raw⟩
        pure (some ⟨region, none, zero, SInfo.scalar, Bool.true⟩)
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandStaticPtrExpr env e
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env a with
      | some p =>
          let b0 := (← expandExpr env b).forgetStorageWidth
          -- A static region offset is accumulated before the natural-address
          -- conversion. Converting each signed addend separately would turn
          -- `cols + base + i` into `base + i.toNat`, which is wrong for i < 0.
          let signedPrefix := match p.offset with
            | `(Op.castIntToNat $_) => Bool.true
            | _ => Bool.false
          if !p.baseOnly && (b0.dtype == .int || signedPrefix) then
            let left ← match p.offset with
              | `(Op.castIntToNat $inner) => pure inner
              | _ => `(Op.castNatToInt $p.offset)
            let right ←
              if b0.dtype == .int then pure b0
              else
                let bNat ← if b0.dtype == .nat then pure b0 else expandNatExpectedExpr env b
                pure { bNat with term := ← `(Op.castNatToInt $bNat.term), dtype := .int }
            let (bc, outShape) ← broadcastTerm p.shape right.shape "pointer offset"
            let total ← `(Op.add NumericDType.int $bc $left $right.term)
            let offset ← pointerOffsetIntAsNat ⟨total, .int, outShape, none, none⟩
            return some ⟨p.region, p.regionDType?, offset.term, outShape, Bool.false⟩
          let b' ←
            if b0.dtype == .int then
              pointerOffsetIntAsNat b0
            else if b0.dtype == .nat then
              pure b0
            else
              expandNatExpectedExpr env b
          if p.baseOnly then
            pure (some ⟨p.region, p.regionDType?, b'.term, b'.shape, Bool.false⟩)
          else
            let (bc, outShape) ← broadcastTerm p.shape b'.shape "pointer offset"
            let off := p.offset
            let bTerm := b'.term
            let nextOff ← `(Op.add NumericDType.nat $bc $off $bTerm)
            pure (some ⟨p.region, p.regionDType?, nextOff, outShape, Bool.false⟩)
      | none => pure (none : Option StaticPtrOut)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      match ← expandStaticPtrExpr env a with
      | some p =>
          -- Accumulate signed offsets before converting the final address:
          -- `base + 2 - 5 + 7` must retain the intermediate negative offset.
          let left ← match p.offset with
            | `(Op.castIntToNat $inner) => pure inner
            | _ => `(Op.castNatToInt $p.offset)
          let right ← expandIntExpectedExpr env b
          ensureAlgorithmOnly "pointer subtraction" right
          let (bc, outShape) ← broadcastTerm p.shape right.shape "pointer subtraction"
          let total ← `(Op.sub NumericDType.int $bc $left $right.term)
          let offset ← pointerOffsetIntAsNat ⟨total, .int, outShape, none, none⟩
          pure (some ⟨p.region, p.regionDType?, offset.term, outShape, Bool.false⟩)
      | none => pure (none : Option StaticPtrOut)
  | _ => pure (none : Option StaticPtrOut)

partial def expandWhereFromCond (env : Env) (c' : EOut)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  -- Python integer scalars acquire the integer tensor branch's carrier.
  -- Do not reinterpret floating literals, explicit casts or runtime values.
  let rec integerLiteral (dtype : DInfo) (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    match e with
    | `(tritonExpr| ($inner:tritonExpr)) => integerLiteral dtype inner
    | `(tritonExpr| $n:num) =>
        if dtype == .nat then return some ⟨← `(Op.constNat $n), .nat, .scalar, none, none⟩
        if dtype == .int || dtype == .int8 || dtype == .int32 then
          return some ⟨← `(Op.constInt $n), .int, .scalar, none, none⟩
        return none
    | `(tritonExpr| -$n:num) =>
        if dtype == .int || dtype == .int8 || dtype == .int32 then
          return some ⟨← `(Op.constInt (-($n : Int))), .int, .scalar, none, none⟩
        return none
    | _ => return none
  let a' := (← integerLiteral b'.dtype a).getD a'
  let b' := (← integerLiteral a'.dtype b).getD b'
  let (a', b') := if a'.dtype == b'.dtype then (a', b')
    else (a'.forgetStorageWidth, b'.forgetStorageWidth)
  -- Triton 3.5.1 casts integer conditions to int1 with `value != 0`.
  -- Keep the original integer load and compare in its own dtype.
  let c' ← match c'.dtype with
    | .int | .int8 | .int32 => do
        ensureAlgorithmOnly "tl.where integer condition" c'
        let zero ← coerceShape (← `(Op.constInt 0)) SInfo.scalar c'.shape "tl.where integer condition"
        let (bc, _) ← broadcastTerm c'.shape c'.shape "tl.where integer condition"
        pure ⟨← `(Op.ne ComparableDType.int $bc $c'.term $zero),
          .bool, c'.shape, none, none⟩
    | .nat => do
        ensureAlgorithmOnly "tl.where integer condition" c'
        let zero ← coerceShape (← `(Op.constNat 0)) SInfo.scalar c'.shape "tl.where integer condition"
        let (bc, _) ← broadcastTerm c'.shape c'.shape "tl.where integer condition"
        pure ⟨← `(Op.ne ComparableDType.nat $bc $c'.term $zero),
          .bool, c'.shape, none, none⟩
    | _ => pure c'
  ensureDType .bool c'.dtype "tl.where condition"
  let expandNatToIntBranch (e : TSyntax `tritonExpr) (out : EOut) : MacroM EOut := do
    match methodCast? e with
    | some (_inner, dt) =>
        let dst ← expandDType dt
        if out.dtype == .nat && dst == .int then
          pure ⟨← `(Op.castNatToInt $out.term), .int, out.shape, none, none⟩
        else
          pure out
    | none => pure out
  let (a', b') ←
    if a'.dtype == b'.dtype then
      pure (a', b')
    else
      if a'.dtype == .nat && b'.dtype == .int then
        pure (⟨← `(Op.castNatToInt $a'.term), .int, a'.shape,
          a'.computeTerm, a'.computeDType?⟩, b')
      else if a'.dtype == .int && b'.dtype == .nat then
        pure (a', ⟨← `(Op.castNatToInt $b'.term), .int, b'.shape,
          b'.computeTerm, b'.computeDType?⟩)
      else do
        let aCast ← expandNatToIntBranch a a'
        let bCast ← expandNatToIntBranch b b'
        pure (aCast, bCast)
  let floating : DInfo → Bool
    | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ => Bool.true
    | _ => Bool.false
  let (a', b') ← if floating a'.dtype && floating b'.dtype then
    coerceRealArithOperands "tl.where" a' b'
    else pure (a', b')
  unless a'.dtype == b'.dtype do
    Macro.throwError "tl.where: branch dtype mismatch"
  let (_, valueShape) ← broadcastTerm a'.shape b'.shape "tl.where branches"
  let (_, target) ← broadcastTerm c'.shape valueShape "tl.where condition"
  let cTerm ← coerceShape c'.term c'.shape target "tl.where condition"
  let aTerm ← coerceShape a'.term a'.shape target "tl.where then-branch"
  let bTerm ← coerceShape b'.term b'.shape target "tl.where else-branch"
  let algTerm ← `(Op.where $cTerm $aTerm $bTerm)
  let (computeTerm, precision) ← floatingComputeArith? "tl.where" algTerm a' b'
  pure ⟨algTerm, a'.dtype, target, computeTerm, precision⟩

partial def expandIteFromCond (env : Env) (c' : EOut)
    (a b : TSyntax `tritonExpr) : MacroM EOut := do
  let a' ← expandExpr env a
  let b' ← expandExpr env b
  let (a', b') := if a'.dtype == b'.dtype then (a', b')
    else (a'.forgetStorageWidth, b'.forgetStorageWidth)
  ensureDType .bool c'.dtype "conditional expression condition"
  ensureShape SInfo.scalar c'.shape "conditional expression condition"
  let expandNatToIntBranch (e : TSyntax `tritonExpr) (out : EOut) : MacroM EOut := do
    match methodCast? e with
    | some (_inner, dt) =>
        let dst ← expandDType dt
        if out.dtype == .nat && dst == .int then
          pure ⟨← `(Op.castNatToInt $out.term), .int, out.shape, none, none⟩
        else
          pure out
    | none => pure out
  let (a', b') ←
    if a'.dtype == b'.dtype then
      pure (a', b')
    else
      if a'.dtype == .nat && b'.dtype == .int then
        pure (⟨← `(Op.castNatToInt $a'.term), .int, a'.shape,
          a'.computeTerm, a'.computeDType?⟩, b')
      else if a'.dtype == .int && b'.dtype == .nat then
        pure (a', ⟨← `(Op.castNatToInt $b'.term), .int, b'.shape,
          b'.computeTerm, b'.computeDType?⟩)
      else do
        let aCast ← expandNatToIntBranch a a'
        let bCast ← expandNatToIntBranch b b'
        pure (aCast, bCast)
  unless a'.dtype == b'.dtype do
    Macro.throwError "conditional expression: branch dtype mismatch"
  let (_, target) ← broadcastTerm a'.shape b'.shape "conditional expression branches"
  let aTerm ← coerceShape a'.term a'.shape target "conditional expression then-branch"
  let bTerm ← coerceShape b'.term b'.shape target "conditional expression else-branch"
  pure ⟨← `(Op.ite $c'.term $aTerm $bTerm), a'.dtype, target, none, none⟩

partial def expandBoolCondition (env : Env) (cond : TSyntax `tritonExpr) :
    MacroM EOut := do
  match cond with
  | `(tritonExpr| $a:ident == $b:str) =>
      if env.any (fun entry => entry.1 == a.getId.toString) then
        expandExpr env cond
      else
        let lhs : TSyntax `term := ⟨a.raw⟩
        let rhs : TSyntax `term := ⟨b.raw⟩
        pure ⟨← `(Op.constBool ($lhs == $rhs)), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| $a:ident != $b:str) =>
      if env.any (fun entry => entry.1 == a.getId.toString) then
        expandExpr env cond
      else
        let lhs : TSyntax `term := ⟨a.raw⟩
        let rhs : TSyntax `term := ⟨b.raw⟩
        pure ⟨← `(Op.constBool ($lhs != $rhs)), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| $b:ident) =>
      if env.any (fun entry => entry.1 == b.getId.toString) then
        expandExpr env cond
      else
        let boolTerm : TSyntax `term := ⟨b.raw⟩
        pure ⟨← `(Op.constBool $boolTerm), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| not $b:ident) =>
      if env.any (fun entry => entry.1 == b.getId.toString) then
        expandExpr env cond
      else
        let boolTerm : TSyntax `term := ⟨b.raw⟩
        pure ⟨← `(Op.boolNot (Op.constBool $boolTerm)), .bool, SInfo.scalar, none, none⟩
  | _ =>
      expandExpr env cond

partial def mergeBranchEnv (base thenEnv elseEnv : Env) : Env :=
  let baseNames := base.map (fun entry => entry.1)
  let thenNew := thenEnv.filter (fun entry => !(baseNames.contains entry.1))
  let elseNew := elseEnv.filter (fun entry => !(baseNames.contains entry.1))
  let mergedNew :=
    thenNew.filterMap (fun t =>
      let (tn, td, ts, tc) := t
      match elseNew.find? (fun e => e.1 == tn) with
      | some e =>
          let (_, ed, es, ec) := e
          if td == ed && ts.eq es && tc == ec then some t
          -- A default floating branch and an explicitly fp32 branch share
          -- the real register carrier. Retain the joined fp32 compute tag.
          else if td == .real && ed == .real && ts.eq es &&
              ((tc == none && ec == some .fp32) || (tc == some .fp32 && ec == none)) then
            some (tn, td, ts, some .fp32)
          else none
      | none => none)
  if mergedNew.isEmpty then
    thenNew ++ base
  else
    mergedNew ++ base

partial def staticBoolCond? (env : Env) (cond : TSyntax `tritonExpr) : Bool :=
  match cond with
  | `(tritonExpr| $a:ident == $_:str) =>
      !env.any (fun entry => entry.1 == a.getId.toString)
  | `(tritonExpr| $a:ident != $_:str) =>
      !env.any (fun entry => entry.1 == a.getId.toString)
  | `(tritonExpr| $b:ident) =>
      !env.any (fun entry => entry.1 == b.getId.toString)
  | `(tritonExpr| not $e:tritonExpr) =>
      staticBoolCond? env e
  | _ => Bool.false


private partial def staticBoolAtom? (env : Env) (cond : TSyntax `tritonExpr) :
    Option ((Name × Option String) × Bool) :=
  match cond with
  | `(tritonExpr| $a:ident == $b:str) =>
      if env.any (fun entry => entry.1 == a.getId.toString) then none
      else some ((a.getId, some b.getString), Bool.true)
  | `(tritonExpr| $a:ident != $b:str) =>
      if env.any (fun entry => entry.1 == a.getId.toString) then none
      else some ((a.getId, some b.getString), Bool.false)
  | `(tritonExpr| $b:ident) =>
      if env.any (fun entry => entry.1 == b.getId.toString) then none
      else some ((b.getId, none), Bool.true)
  | `(tritonExpr| not $e:tritonExpr) =>
      (staticBoolAtom? env e).map (fun (atom, positive) => (atom, !positive))
  | _ => none

private partial def knownStaticBool? (env : Env) (facts : StaticBoolFacts)
    (cond : TSyntax `tritonExpr) : Option Bool := do
  let (atom, positive) ← staticBoolAtom? env cond
  let (_, value) ← facts.find? (fun fact => fact.1 == atom)
  pure (value == positive)

private partial def withStaticBool (env : Env) (facts : StaticBoolFacts)
    (cond : TSyntax `tritonExpr) (value : Bool) : StaticBoolFacts :=
  match staticBoolAtom? env cond with
  | some (atom, positive) => (atom, value == positive) :: facts
  | none => facts

partial def mergeStaticBranchEnv (base thenEnv elseEnv : Env) : Env :=
  -- Retain rebinding history as well as newly introduced names. An enclosing
  -- constexpr branch must see a nested dtype change even when the name was
  -- already present before either branch. Divergent continuations are expanded
  -- separately by `expandStmts`, so this is not a runtime dtype join.
  thenEnv.take (thenEnv.length - base.length) ++
    elseEnv.take (elseEnv.length - base.length) ++ base

private partial def sameBranchBindingType
    (a b : String × DInfo × SInfo × Option DInfo) : Bool :=
  let (_, ad, ash, ac) := a
  let (_, bd, bsh, bc) := b
  ad == bd && ash.eq bsh &&
    (ac == bc || (ad == .real &&
      ((ac == none && bc == some .fp32) || (ac == some .fp32 && bc == none))))

private partial def staticBranchChangesType (base thenEnv elseEnv : Env) : Bool :=
  let bindings := thenEnv.take (thenEnv.length - base.length) ++
    elseEnv.take (elseEnv.length - base.length)
  bindings.any fun binding =>
    (match base.find? (fun old => old.1 == binding.1) with
     | some old => !sameBranchBindingType old binding
     | none => Bool.false) ||
    bindings.any (fun other => other.1 == binding.1 &&
      !sameBranchBindingType binding other)

/-- Intermediate assignments within a straight-line arm are not alternative
exit types. For arms containing control flow we retain the conservative history:
it can encode distinct types at the exits of nested constexpr branches. -/
private partial def staticBranchExitEnv (base branch : Env)
    (stmts : List (TSyntax `tritonStmt)) : Env :=
  let linear := stmts.all fun st => match st with
    | `(tritonStmt| $_:ident = $_:tritonExpr) => Bool.true
    | `(tritonStmt| $_:ident := $_:tritonExpr) => Bool.true
    | _ => Bool.false
  if !linear then branch else
    let fresh := branch.take (branch.length - base.length)
    let rec latest : Env → Env
      | [] => []
      | binding :: rest => binding :: latest (rest.filter (fun other => other.1 != binding.1))
    latest fresh ++ base

partial def natOffsetsWithBaseAdd (env : Env) (ctx : String)
    (offsets strides : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × Option (TSyntax `term)) := do
  unless offsets.size == strides.size do
    Macro.throwError (ctx ++ ": offsets and strides must have the same rank")
  let mut staticOffsets : Array (TSyntax `term) := #[]
  let mut dynamicAdds : Array (TSyntax `term) := #[]
  for i in [:offsets.size] do
    let off := offsets[i]!
    let strideTerm ← natDimTerm (ctx ++ " strides") strides[i]!
    match off with
    | `(tritonExpr| $n:num) =>
        staticOffsets := staticOffsets.push (⟨n.raw⟩ : TSyntax `term)
    | `(tritonExpr| $($t:term)) =>
        staticOffsets := staticOffsets.push (← `(($t : Nat)))
    | `(tritonExpr| $id:ident) =>
        if env.any (fun entry => entry.1 == id.getId.toString) then
          staticOffsets := staticOffsets.push (← `((0 : Nat)))
          let off' ← expandNatExpectedExpr env off
          ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
          dynamicAdds := dynamicAdds.push
            (← `(Op.mul NumericDType.nat Broadcast.nil $off'.term (Op.constNat $strideTerm)))
        else
          let t : TSyntax `term := ⟨id.raw⟩
          staticOffsets := staticOffsets.push (← `(($t : Nat)))
    | _ =>
        staticOffsets := staticOffsets.push (← `((0 : Nat)))
        let off' ← expandNatExpectedExpr env off
        ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
        dynamicAdds := dynamicAdds.push
          (← `(Op.mul NumericDType.nat Broadcast.nil $off'.term (Op.constNat $strideTerm)))
  let rec listTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Nat))
    | d :: rest => do
        let tail ← listTerm rest
        `($d :: $tail)
  let rec sumOps : List (TSyntax `term) → MacroM (Option (TSyntax `term))
    | [] => pure none
    | [x] => pure (some x)
    | x :: xs => do
        match ← sumOps xs with
        | none => pure (some x)
        | some rest =>
            pure (some (← `(Op.add NumericDType.nat Broadcast.nil $x $rest)))
  pure (← listTerm staticOffsets.toList, ← sumOps dynamicAdds.toList)

/-- Preserve the static advance AST for constant coordinates; runtime
coordinates use signed scalar operations and retain lookup failure. -/
partial def blockPtrDeltas (env : Env) (offsets : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × TSyntax `term × Bool) := do
  let mut constants : Array (TSyntax `term) := #[]
  let mut ops : Array (TSyntax `term) := #[]
  let mut dynamic := Bool.false
  for off in offsets do
    let staticTerm? ← match off with
      | `(tritonExpr| $n:num) => pure (some (← `(($n : Int))))
      | `(tritonExpr| $(($t:term : Int))) => pure (some (← `(($t : Int))))
      | `(tritonExpr| $($t:term)) => pure (some (← `((($t : Nat) : Int))))
      | `(tritonExpr| $id:ident) =>
          if env.any (fun entry => entry.1 == id.getId.toString ||
              entry.1 == id.getId.eraseMacroScopes.toString) then pure none
          else
            let t : TSyntax `term := ⟨id.raw⟩
            pure (some (← `((($t : Nat) : Int))))
      | _ => pure none
    match staticTerm? with
    | some t =>
        constants := constants.push t
        ops := ops.push (← `(Op.constInt $t))
    | none =>
        dynamic := Bool.true
        constants := constants.push (← `((0 : Int)))
        let value ← expandIntExpectedExpr env off
        ensureShape SInfo.scalar value.shape "tl.advance offsets"
        ops := ops.push value.term
  let rec intList : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Int))
    | x :: xs => do let tail ← intList xs; `($x :: $tail)
  let rec opList : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List (Op .int [])))
    | x :: xs => do let tail ← opList xs; `($x :: $tail)
  pure (← intList constants.toList, ← opList ops.toList, dynamic)

partial def natBlockPtrOffsets (env : Env) (ctx : String)
    (offsets : Array (TSyntax `tritonExpr)) :
    MacroM (TSyntax `term × TSyntax `term × Bool) := do
  let signedNames := env.filterMap fun entry =>
    if entry.2.1 == DInfo.int || entry.2.1 == DInfo.int8 ||
        entry.2.1 == DInfo.int32 then some entry.1 else none
  let expandOffset (off : TSyntax `tritonExpr) : MacroM EOut := do
    if Inference.isSignedExpr signedNames off then
      let value ← expandIntExpectedExpr env off
      -- Evaluate signed arithmetic before projecting the complete address.
      return ⟨← `(Op.castIntToNat $value.term), .nat, value.shape, none, none⟩
    expandNatExpectedExpr env off
  let mut staticOffsets : Array (TSyntax `term) := #[]
  let mut offsetOps : Array (TSyntax `term) := #[]
  let mut hasDynamic := Bool.false
  for off in offsets do
    match off with
    | `(tritonExpr| $n:num) =>
        staticOffsets := staticOffsets.push (⟨n.raw⟩ : TSyntax `term)
        offsetOps := offsetOps.push (← `(Op.constNat $n))
    | `(tritonExpr| $($t:term)) =>
        staticOffsets := staticOffsets.push (← `(($t : Nat)))
        offsetOps := offsetOps.push (← `(Op.constNat ($t : Nat)))
    | `(tritonExpr| $id:ident) =>
        if env.any (fun entry => entry.1 == id.getId.toString) then
          hasDynamic := Bool.true
          staticOffsets := staticOffsets.push (← `((0 : Nat)))
          let off' ← expandOffset off
          ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
          offsetOps := offsetOps.push off'.term
        else
          let t : TSyntax `term := ⟨id.raw⟩
          staticOffsets := staticOffsets.push (← `(($t : Nat)))
          offsetOps := offsetOps.push (← `(Op.constNat ($t : Nat)))
    | _ =>
        hasDynamic := Bool.true
        staticOffsets := staticOffsets.push (← `((0 : Nat)))
        let off' ← expandOffset off
        ensureShape SInfo.scalar off'.shape (ctx ++ " offsets")
        offsetOps := offsetOps.push off'.term
  let rec natListTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List Nat))
    | d :: rest => do
        let tail ← natListTerm rest
        `($d :: $tail)
  let rec opListTerm : List (TSyntax `term) → MacroM (TSyntax `term)
    | [] => `(([] : List (Op .nat [])))
    | d :: rest => do
        let tail ← opListTerm rest
        `($d :: $tail)
  pure (← natListTerm staticOffsets.toList, ← opListTerm offsetOps.toList, hasDynamic)

partial def expandExpr (env : Env) (stx : TSyntax `tritonExpr) : MacroM EOut := do
  if let `(tritonExpr| tl.inline_asm_elementwise($assembly:str $[, $kwargs:tritonInlineAsmKwarg]*)) := stx then
    let (a,b) ← inlineAddU32Arguments assembly kwargs
    let left ← expandExpr env a
    let right ← expandExpr env b
    unless [.int,.int32].contains left.dtype && [.int,.int32].contains right.dtype do
      Macro.throwError "inline ASM add.u32: operands must use integer registers"
    -- Each operand is a register read, used once. Preserve the exact prior
    -- modulo32/signed32 primitive; this is not floating addition or an FMA.
    return ← expandExpr env (← `(tritonExpr| (($a:tritonExpr + $b:tritonExpr).to(tl.uint32)).to(tl.int32)))
  -- The source's explicit output-pointer cast can be Bool -> Bool. Preserve
  -- its actual operand instead of treating the register as a numeric load.
  if let some (e, dt) := methodCast? stx then
    if (← expandDType dt) == .bool then
      let e ← if e.raw.isIdent then do
        let i : TSyntax `ident := ⟨e.raw⟩
        `(tritonExpr| $i:ident)
      else pure e
      let original ← expandExpr env e
      if original.dtype == .bool then return original
  if let some (e, dt) := methodBitcast? stx then
    let e ← if e.raw.isIdent then do
      let i : TSyntax `ident := ⟨e.raw⟩
      `(tritonExpr| $i:ident)
    else pure e
    return ← expandExpr env (← `(tritonExpr| tl.bitcast($e:tritonExpr, $dt:tritonDType)))
  if stx.raw.getKind == ``tritonDottedIdentMethodCast then
    let args := stx.raw.getArgs
    if h : args.size = 4 then
      let i : TSyntax `ident := ⟨args[0]⟩
      let dt : TSyntax `tritonDType := ⟨args[2]⟩
      match dottedToBaseName? i with
      | some name =>
          let e' ← expandIdentRefByName env name
          if let some cast ← machineIntegerCast? dt e' then return cast
          let dst ← expandDType dt
          match dst with
          | .bool =>
              Macro.throwError "tl.cast: Bool casts are only supported directly on tl.load"
          | .nat =>
              match e'.dtype with
              | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
                  Macro.throwError "float-to-unsigned cast requires an explicit supported target width"
              | .int | .int8 | .int32 =>
                  pure ⟨← `(Op.castIntToNat $e'.term), .nat, e'.shape, none, none⟩
              | _ => pure e'
          | .int =>
              match e'.dtype with
              | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
                  Macro.throwError "float-to-signed cast requires an explicit supported target width"
              | .nat =>
                  pure ⟨← `(Op.castNatToInt $e'.term), .int, e'.shape, none, none⟩
              | _ => pure (e'.integerIdentityCast dt)
          | .fp32 =>
              explicitFloatCast .fp32 e'
          | .real =>
              explicitFloatCast .real e'
          | _ =>
              explicitFloatCast dst e'
      | none => Macro.throwUnsupported
    else
      Macro.throwUnsupported
  else
  if stx.raw.getKind == ``tritonMethodCastElementTy ||
      stx.raw.getKind == ``tritonMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastDTypeIdentSpaced ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdent ||
      stx.raw.getKind == ``tritonIdentMethodCastElementTyIdentSpaced then
    let args := stx.raw.getArgs
    if h : 0 < args.size then
      let e : TSyntax `tritonExpr := ⟨args[0]⟩
      let target := (← methodCastTargetDType? stx).orElse
        (fun _ => methodCastRegisterDType? env stx)
      match target with
      | some .int =>
          let original ← expandExpr env e
          if methodCastTargetName? stx == some `tl.int32 then
            if let some cast ← machineIntegerCast? (← `(tritonDType| tl.int32)) original then
              return cast
          expandIntExpectedExpr env e
      | some .nat => expandNatExpectedExpr env e
      | some dst =>
          match dst with
          | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
              explicitFloatCast dst (← expandExpr env e)
          | _ => expandExpr env e
      | none => expandExpr env e
    else
      Macro.throwUnsupported
  else
  match roundCast? stx with
  | some (e, dt) =>
      -- `.round_to(tl.floatXX)`: always the FloatDType rounding channel —
      -- an explicit quantization event (`Op.castFloat src dst`), never the
      -- bit-accurate compute channel `.to(tl.float32)` routes fp32 to.
      let e' ← expandExpr env e
      let dst ← expandDType dt
      match dst with
      | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 =>
          let srcProof ← e'.dtype.floatProof
          let dstProof ← dst.floatProof
          pure ⟨← `(Op.castFloat $srcProof $dstProof $e'.term), dst, e'.shape,
            none, none⟩
      | _ =>
          Macro.throwError
            "round_to: only fp32/fp16/bf16/float8 name a rounding grid (use .to(...) for working-precision casts)"
  | none =>
  match methodCast? stx with
  | some (e, dt) =>
      let dst ← expandDType dt
      match dst with
      | .bool =>
          let rec expandBoolLoadCast (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
            match e with
            | `(tritonExpr| ($inner:tritonExpr)) =>
                expandBoolLoadCast inner
            | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
                pure (some (← expandLoad expandExpr expandStaticPtrExpr env p kwargs (some .bool) (some mask)))
            | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
                pure (some (← expandLoad expandExpr expandStaticPtrExpr env p kwargs (some .bool)))
            | _ =>
                pure none
          match ← expandBoolLoadCast e with
          | some out => pure out
          | none =>
              Macro.throwError "tl.cast: Bool casts are only supported directly on tl.load"
      | _ =>
          let e' ← expandExpr env e
          if let some cast ← machineIntegerCast? dt e' then return cast
          match e'.dtype, dst with
          | .real, .nat | .fp32, .nat | .fp16, .nat | .bf16, .nat
          | .f8e4, .nat | .f8e5, .nat
          | .floatVar _, .nat =>
              Macro.throwError "float-to-unsigned cast requires an explicit supported target width"
          | .int, .nat | .int8, .nat | .int32, .nat =>
              pure ⟨← `(Op.castIntToNat $e'.term), .nat, e'.shape, none, none⟩
          | .real, .int | .fp32, .int | .fp16, .int | .bf16, .int
          | .f8e4, .int | .f8e5, .int
          | .floatVar _, .int =>
              Macro.throwError "float-to-signed cast requires an explicit supported target width"
          | .nat, .int =>
              pure ⟨← `(Op.castNatToInt $e'.term), .int, e'.shape, none, none⟩
          | _, .int =>
              pure (e'.integerIdentityCast dt)
          | _, .fp32 =>
              explicitFloatCast .fp32 e'
          | _, .real =>
              explicitFloatCast .real e'
          | _, _ =>
              explicitFloatCast dst e'
  | none =>
  match stx with
  | `(tritonExpr| $n:num) =>
      -- Bare numeric literals are `ℝ` data constants (e.g. `1` in `1 / s`).
      pure ⟨← `(Op.const $n), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| $n:scientific) =>
      pure ⟨← `(Op.const $n), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| true) =>
      pure ⟨← `(Op.constBool Bool.true), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| false) =>
      pure ⟨← `(Op.constBool Bool.false), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| $i:ident) =>
      let rawName := i.getId.toString
      let userName := i.getId.getString!
      let erasedName := i.getId.eraseMacroScopes.toString
      unless env.any (fun entry =>
          entry.1 == rawName || entry.1 == userName || entry.1 == erasedName) do
        let t : TSyntax `term := ⟨i.raw⟩
        return ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
      let name :=
        if env.any (fun entry => entry.1 == rawName) then rawName
        else if env.any (fun entry => entry.1 == userName) then userName
        else erasedName
      expandIdentRefByName env name
  | `(tritonExpr| $((VeriTile.Triton.DSL.TypedStores.fp32CallValue $value:term : ℝ))) => do
      let some out ← suppliedFP32Literal? stx
        | Macro.throwError "invalid internal FP32 call literal"
      return out
  | `(tritonExpr| $(($t:term : ℝ))) =>
      pure ⟨← `(Op.const $t), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| $(($t:term : Real))) =>
      pure ⟨← `(Op.const $t), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| $(($t:term : Bool))) =>
      pure ⟨← `(Op.constBool $t), .bool, SInfo.scalar, none, none⟩
  | `(tritonExpr| $(($t:term : Nat))) =>
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| $(($t:term : Int))) =>
      pure ⟨← `(Op.constInt $t), .int, SInfo.scalar, none, none⟩
  | `(tritonExpr| $($t:term)) =>
      -- `$(...)` antiquote is the address/size channel: `Nat`.
      -- Data/scalar contexts reinterpret the same surface form through
      -- `expandLeanAntiquoteAs? .real` before this default is used.
      pure ⟨← `(Op.constNat $t), .nat, SInfo.scalar, none, none⟩
  | `(tritonExpr| ($e:tritonExpr)) =>
      expandExpr env e
  | `(tritonExpr| tl.program_id($e:tritonExpr)) =>
      expandProgramId e
  | `(tritonExpr| tl.program_id(axis=$e:tritonExpr)) =>
      expandProgramId e
  | `(tritonExpr| tl.num_programs($e:tritonExpr)) =>
      expandNumPrograms e
  | `(tritonExpr| tl.num_programs(axis=$e:tritonExpr)) =>
      expandNumPrograms e
  | `(tritonExpr| tl.arange($e:tritonExpr)) =>
      expandArange e
  | `(tritonExpr| tl.arange($s:tritonExpr, $e:tritonExpr)) => do
      expandArangeRangeWithEnv env s e
  | `(tritonExpr| tl.__uniform32_profiled($input:tritonExpr)) => do
      expandDirectUniform32 (← expandExpr env input)
  | `(tritonExpr| tl.__unsigned_add_profiled($width:num, $lt:term, $rt:term,
      $left:tritonExpr, $right:tritonExpr)) => do
      unless width.getNat == 32 || width.getNat == 64 do
        Macro.throwError "unsupported internal unsigned arithmetic width"
      let some leftType := RandomProfile.integerTypeTerm? lt |
        Macro.throwError "malformed internal unsigned lhs profile"
      let some rightType := RandomProfile.integerTypeTerm? rt |
        Macro.throwError "malformed internal unsigned rhs profile"
      expandDirectUnsignedArithmetic width.getNat Bool.false
        (← expandRandIntegerExpr env leftType left) (← expandRandIntegerExpr env rightType right)
  | `(tritonExpr| tl.__unsigned_mul_profiled($width:num, $lt:term, $rt:term,
      $left:tritonExpr, $right:tritonExpr)) => do
      unless width.getNat == 32 || width.getNat == 64 do
        Macro.throwError "unsupported internal unsigned arithmetic width"
      let some leftType := RandomProfile.integerTypeTerm? lt |
        Macro.throwError "malformed internal unsigned lhs profile"
      let some rightType := RandomProfile.integerTypeTerm? rt |
        Macro.throwError "malformed internal unsigned rhs profile"
      expandDirectUnsignedArithmetic width.getNat Bool.true
        (← expandRandIntegerExpr env leftType left) (← expandRandIntegerExpr env rightType right)
  | `(tritonExpr| tl.__signed_and64_profiled($rt:term, $left:tritonExpr, $right:tritonExpr)) => do
      let some rightType := RandomProfile.integerTypeTerm? rt |
        Macro.throwError "malformed internal signed bitwise rhs profile"
      let lhs ← expandRandIntegerExpr env .i64 left
      let rhs ← expandRandIntegerExpr env rightType right
      let (bc, shape) ← broadcastTerm lhs.shape rhs.shape "signed int64 bitwise AND"
      return ⟨← `(Op.intBitAnd 64 $bc $lhs.term $rhs.term), .int, shape, none, none⟩
  | `(tritonExpr| tl.__signed_shift64_profiled($left:tritonExpr, $count:num)) => do
      unless count.getNat < 64 do
        Macro.throwError "signed int64 right shift requires a literal count below 64"
      let lhs ← expandRandIntegerExpr env .i64 left
      let (bc, shape) ← broadcastTerm lhs.shape SInfo.scalar "signed int64 right shift"
      return ⟨← `(Op.intShiftRight $bc $lhs.term (Op.constNat $count)), .int, shape, none, none⟩
  | `(tritonExpr| tl.__rand_profiled($cfg:term, $seed:tritonExpr, $offsets:tritonExpr)) => do
      let some (seedType, offsetType) := RandomProfile.configTypes? cfg |
        Macro.throwError "tl.rand: malformed internal source profile"
      let seed' ← expandRandIntegerExpr env seedType seed
      let offsets' ← expandRandIntegerExpr env offsetType offsets
      let term ← if seed'.shape.eq SInfo.scalar then
          `(Op.rand $cfg $seed'.term $offsets'.term)
        else do
          ensureShape offsets'.shape seed'.shape
            "tl.rand: vector seeds require the same shape as offsets"
          `(Op.randTensor $cfg $seed'.term $offsets'.term)
      pure ⟨term, .real, offsets'.shape, some (← fp32ComputeExpr term), some .fp32⟩
  | `(tritonExpr| tl.rand($_:tritonExpr, $_:tritonExpr))
  | `(tritonExpr| tl.rand($_:tritonExpr, $_:tritonExpr, $_:tritonExpr))
  | `(tritonExpr| tl.rand($_:tritonExpr, $_:tritonExpr, n_rounds=$_:tritonExpr)) =>
      Macro.throwError "tl.rand requires source-profile preprocessing in a triton block"
  | `(tritonExpr| tl.exp($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.exp" e'
      let term ← `(Op.exp $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.exp" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| libdevice.exp($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.exp($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "libdevice.exp" e'
      let term ← `(Op.libdeviceExp $eTerm)
      let (ct, cp) ← floatingComputeArith? "libdevice.exp" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| libdevice.log($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.log($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "libdevice.log" e'
      let term ← `(Op.libdeviceLog $eTerm)
      let (ct, cp) ← floatingComputeArith? "libdevice.log" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| libdevice.expm1($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.expm1($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "libdevice.expm1" e'
      let term ← `(Op.libdeviceExpm1 $eTerm)
      let (ct, cp) ← floatingComputeArith? "libdevice.expm1" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| libdevice.log1p($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.log1p($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "libdevice.log1p" e'
      let term ← `(Op.libdeviceLog1p $eTerm)
      let (ct, cp) ← floatingComputeArith? "libdevice.log1p" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.exp2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.exp2" e'
      let term ← `(Op.exp2 $eTerm)
      let (ct, cp) ← floatingComputeArith? "exp2" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.math.exp2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.exp2" e'
      let term ← `(Op.exp2 $eTerm)
      let (ct, cp) ← floatingComputeArith? "exp2" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.extra.cuda.libdevice.pow($e:tritonExpr, $b:tritonExpr)) =>
      expandLibdevicePow expandExpr env e b
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.log" e'
      let term ← `(Op.log $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.log" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.log2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.log2" e'
      let term ← `(Op.log2 $eTerm)
      let (ct, cp) ← floatingComputeArith? "log2" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.math.log2($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.log2" e'
      let term ← `(Op.log2 $eTerm)
      let (ct, cp) ← floatingComputeArith? "log2" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.sigmoid($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sigmoid" e'
      let term ← `(Op.sigmoid $eTerm)
      let (ct, cp) ← floatingComputeArith? "sigmoid" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| silu($e:tritonExpr)) =>
      expandExpr env (← `(tritonExpr| $e * tl.sigmoid($e)))
  | `(tritonExpr| leaky_relu($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "leaky_relu" e'
      let (cmpBc, cmpShape) ← broadcastTerm e'.shape SInfo.scalar "leaky_relu"
      ensureShape e'.shape cmpShape "leaky_relu"
      let (mulBc, mulShape) ← broadcastTerm SInfo.scalar e'.shape "leaky_relu"
      ensureShape e'.shape mulShape "leaky_relu"
      let zero ← `(Op.const 0)
      let slope ← `(Op.const (0.01 : ℝ))
      let cond ← `(Op.ge ComparableDType.real $cmpBc $eTerm $zero)
      let scaled ← `(Op.mul NumericDType.real $mulBc $slope $eTerm)
      pure ⟨← `(Op.where $cond $eTerm $scaled), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.fma($a:tritonExpr, $b:tritonExpr, $c:tritonExpr)) =>
      expandFma expandExpr env a b c
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sqrt" e'
      let term ← `(Op.sqrt $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.sqrt" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.math.rsqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.rsqrt" e'
      let term ← `(Op.rsqrt $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.math.rsqrt" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.rsqrt($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.rsqrt" e'
      let term ← `(Op.rsqrt $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.rsqrt" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.tanh" e'
      let term ← `(Op.tanh $eTerm)
      let (ct, cp) ← floatingComputeArith? "tanh" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  -- The corpus alias triton_tanh denotes libdevice.tanh. Keep its backend
  -- separate from tl.tanh, including the propagated compute precision.
  | `(tritonExpr| triton_tanh($e:tritonExpr))
  | `(tritonExpr| libdevice.tanh($e:tritonExpr))
  | `(tritonExpr| tl.extra.cuda.libdevice.tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "libdevice.tanh" e'
      let term ← `(Op.libdeviceTanh $eTerm)
      let (ct, cp) ← floatingComputeArith? "libdevice.tanh" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tanh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tanh" e'
      let term ← `(Op.tanh $eTerm)
      let (ct, cp) ← floatingComputeArith? "tanh" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.sin($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sin" e'
      let term ← `(Op.sin $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.sin" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.math.sin($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.math.sin" e'
      let term ← `(Op.sin $eTerm)
      let (ct, cp) ← floatingComputeArith? "tl.math.sin" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.cos($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.cos" e'
      let term ← `(Op.cos $eTerm)
      let (ct, cp) ← floatingComputeArith? "cos" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.tan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.tan" e'
      let term ← `(Op.tan $eTerm)
      let (ct, cp) ← floatingComputeArith? "tan" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.atan($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.atan" e'
      let term ← `(Op.atan $eTerm)
      let (ct, cp) ← floatingComputeArith? "atan" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.cosh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.cosh" e'
      let term ← `(Op.cosh $eTerm)
      let (ct, cp) ← floatingComputeArith? "cosh" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.sinh($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.sinh" e'
      let term ← `(Op.sinh $eTerm)
      let (ct, cp) ← floatingComputeArith? "sinh" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.erf($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.erf" e'
      let term ← `(Op.erf $eTerm)
      let (ct, cp) ← floatingComputeArith? "erf" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.extra.cuda.libdevice.erf($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let eTerm ← realMathTerm "tl.extra.cuda.libdevice.erf" e'
      let term ← `(Op.erf $eTerm)
      let (ct, cp) ← floatingComputeArith? "erf" term e' e'
      pure ⟨term, .real, e'.shape, ct, cp⟩
  | `(tritonExpr| tl.abs($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let dtype ← e'.dtype.floatProof
      let term ← `(Op.absFloat $dtype $e'.term)
      match e'.computeDType? with
      | some .fp32 =>
          let child ← e'.scopedFloatingTerm
          let fp ← `(Op.absFloat $dtype $child)
          pure ⟨term, e'.dtype, e'.shape, some (← fp32ComputeExpr fp), some .fp32⟩
      | some .real =>
          let child ← e'.scopedFloatingTerm
          let fp ← `(Op.absFloat $dtype $child)
          pure ⟨term, e'.dtype, e'.shape, some (← fp64ComputeExpr fp), some .real⟩
      | _ =>
          if e'.computeTerm.isSome then
            let child ← e'.floatingTerm
            let fp ← `(Op.absFloat $dtype $child)
            pure ⟨term, e'.dtype, e'.shape, some (← `(ComputeExpr.alg $fp)), none⟩
          else pure ⟨term, e'.dtype, e'.shape, none, none⟩
  | `(tritonExpr| tl.logical_and($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin expandExpr env "tl.logical_and" (← `(Op.boolAnd)) a b
  | `(tritonExpr| tl.logical_or($a:tritonExpr, $b:tritonExpr)) => do
      expandBoolBin expandExpr env "tl.logical_or" (← `(Op.boolOr)) a b
  | `(tritonExpr| tl.logical_not($a:tritonExpr)) => do
      expandBoolNot expandExpr env "tl.logical_not" a
  | `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) => do
      expandCdiv expandExpr env a b
  | `(tritonExpr| tl.max($e:tritonExpr, $n:num)) => do
      expandReduceMax expandExpr env e #[← `(tritonReduceKwarg| $n:num)]
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      match b with
      | `(tritonExpr| $n:num) =>
          expandReduceMax expandExpr env a #[← `(tritonReduceKwarg| $n:num)]
      | _ =>
          let a' ← expandExpr env a
          let b' ← expandExpr env b
          ensureDType .real a'.dtype "tl.max"
          ensureDType .real b'.dtype "tl.max"
          let (bc, outShape) ← broadcastTerm a'.shape b'.shape "tl.max"
          pure ⟨← `(Op.max2 $bc $a'.term $b'.term), .real, outShape, none, none⟩
  | `(tritonExpr| tl.maximum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax expandExpr env "tl.maximum" (← `(Op.gt)) a b
  | `(tritonExpr| tl.minimum($a:tritonExpr, float ($arg:term))) => do
      -- Keep this operation in the FP AST. Only its real interpretation is
      -- identity; no numerical FP rewrite is inferred from the literal.
      match arg with
      | `(term| $s:str) =>
          if s.getString == "inf" then
            let a' ← expandExpr env a
            -- This node currently models real-carrier fp32/fp64 operands.
            -- Do not silently widen lower-precision minimum operations.
            ensureDType .real a'.dtype "tl.minimum(..., +inf): explicitly cast to fp32 or fp64 first"
            ensureComputeArithComposable "tl.minimum(..., +inf)" a'
            let aTerm ← realMathTerm "tl.minimum(..., +inf)" a'
            let term ← `(Op.minPosInf $aTerm)
            let (ct, cp) ← floatingComputeArith? "tl.minimum(..., +inf)" term a' a'
            pure ⟨term, .real, a'.shape, ct, cp⟩
          else
            expandMinMax expandExpr env "tl.minimum" (← `(Op.lt)) a
              (← `(tritonExpr| float ($arg:term)))
      | _ => Macro.throwError "tl.minimum: expected the literal float(\"inf\")"
  | `(tritonExpr| tl.minimum($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax expandExpr env "tl.minimum" (← `(Op.lt)) a b
  | `(tritonExpr| min($a:tritonExpr, $b:tritonExpr)) => do
      expandMinMax expandExpr env "min" (← `(Op.lt)) a b
  | `(tritonExpr| tl.cumsum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.cumsum" (← `(ScanOp.sum)) e kwargs
  | `(tritonExpr| tl.cumprod($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.cumprod" (← `(ScanOp.prod)) e kwargs
  | `(tritonExpr| tl.associative_scan($e:tritonExpr, $op:tritonScanOp $[, $kwargs:tritonReduceKwarg]*)) => do
      expandScan expandExpr env "tl.associative_scan" (← expandScanOp op) e kwargs
  | `(tritonExpr| tl.argmax($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce expandExpr env "tl.argmax" (← `(Op.argMax)) e kwargs
  | `(tritonExpr| tl.argmin($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandArgReduce expandExpr env "tl.argmin" (← `(Op.argMin)) e kwargs
  | `(tritonExpr| tl.sort($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandSort expandExpr env e kwargs
  | `(tritonExpr| tl.sum($e:tritonExpr, axis=1, tree=$($tree:term), keep_dims=true)) => do
      let input ← expandExpr env e
      ensureDType .real input.dtype "ordered tl.sum"
      unless input.computeDType? == some .fp32 do
        Macro.throwError "ordered tl.sum requires explicit FP32 input precision"
      let .dims [row, _] := input.shape
        | Macro.throwError "ordered tl.sum requires a [1, N] tile"
      unless termIsOne row do
        Macro.throwError "ordered tl.sum requires a [1, N] tile"
      let inputTerm ← input.scopedFloatingTerm
      let term ← `(Op.full [1,1] (WelfordSource.sumExpression $inputTerm $tree))
      pure ⟨term, .real, .dims [← `(1), ← `(1)],
        some (← fp32ComputeExpr term), some .fp32⟩
  | `(tritonExpr| tl.sum($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce expandExpr env "tl.sum" (← `(Op.reduceSum)) e kwargs
  | `(tritonExpr| tl.min($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduce expandExpr env "tl.min" (← `(Op.reduceMin)) e kwargs
  | `(tritonExpr| tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      expandReduceMax expandExpr env e kwargs
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr env e
      match e'.dtype with
      | .int | .int8 | .int32 => pure ⟨← `(Op.intToReal $e'.term), .real, e'.shape, none, none⟩
      | _ =>
          ensureDType .nat e'.dtype "tl.toReal"
          pure ⟨← `(Op.natToReal $e'.term), .real, e'.shape, none, none⟩
  | `(tritonExpr| tl.multiple_of($e:tritonExpr, $_align:tritonExpr)) => do
      expandExpr env e
  | `(tritonExpr| tl.max_contiguous($e:tritonExpr, $_align:tritonExpr)) => do
      expandExpr env e
  | `(tritonExpr| tl.bitcast($e:tritonExpr, $dt:tritonDType)) => do
      if (← expandDType dt) == .f8e5 then
        let e' ← expandExpr env e
        unless e'.dtype == .int8 do
          Macro.throwError "tl.bitcast(..., tl.float8e5): requires a known int8 source width"
        ensureAlgorithmOnly "int8-to-float8e5 bitcast" e'
        return ⟨← `(Op.decodeFloat8E5 FloatDType.f8e5 $e'.term),
          .f8e5, e'.shape, none, none⟩
      let dst ← expandComputeDType dt
      let dstTerm ← dst.term
      match e with
      | `(tritonExpr| $n:num) =>
          let bits := n.getNat
          unless bits < 2^32 do
            Macro.throwError "tl.bitcast(...): literal does not fit 32 bits"
          if dst == .fp32 then
            validateFp32BitsForAlg bits
          let payload ← computeLiteralPayloadTerm .uint32 bits
          let op ←
            `(ComputeOp.bitcast ComputeDType.uint32 $dstTerm rfl
                (ComputeOp.const $payload))
          let algTerm ← computeLiteralAlgTerm dst bits
          pure ⟨algTerm, dst.algDType, SInfo.scalar,
            some (← `(ComputeExpr.compute $op)), none⟩
      | _ =>
          let e' ← expandExpr env e
          let src ← inferComputeSourceDType "tl.bitcast" e'.dtype
          let srcTerm ← src.term
          let algTerm ← dst.placeholderAlgTerm e'.shape
          let op ←
            `(ComputeOp.bitcast $srcTerm $dstTerm rfl
                (ComputeOp.alg $srcTerm $e'.term))
          pure ⟨algTerm, dst.algDType, e'.shape,
            some (← `(ComputeExpr.compute $op)), none⟩
  | `(tritonExpr| tl.cast($e:tritonExpr, $dt:tritonDType)) => do
      -- A Python integer literal is an integer source value, including when
      -- passed directly to tl.cast. Do not route it through an FP conversion.
      let integerTarget := (unsignedCastWidth? dt).isSome || signedMachineCast dt ||
        (match dt with | `(tritonDType| tl.int64) => Bool.true | _ => Bool.false)
      let e' ← if integerTarget then
          match e with
          | `(tritonExpr| $_:num) => expandNatExpectedExpr env e
          | `(tritonExpr| -$_:num) => expandIntExpectedExpr env e
          | _ => expandExpr env e
        else expandExpr env e
      if let some cast ← machineIntegerCast? dt e' then return cast
      let dst ← expandDType dt
      match dst with
      | .nat =>
          match e'.dtype with
          | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
              Macro.throwError "float-to-unsigned cast requires an explicit supported target width"
          | .int | .int8 | .int32 =>
              pure ⟨← `(Op.castIntToNat $e'.term), .nat, e'.shape, none, none⟩
          | _ => pure e'
      | .int =>
          match e'.dtype with
          | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
              Macro.throwError "float-to-signed cast requires an explicit supported target width"
          | .nat =>
              pure ⟨← `(Op.castNatToInt $e'.term), .int, e'.shape, none, none⟩
          | _ => pure (e'.integerIdentityCast dt)
      | .fp32 =>
          explicitFloatCast .fp32 e'
      | .real =>
          explicitFloatCast .real e'
      | _ =>
          explicitFloatCast dst e'
  | `(tritonExpr| tl.extra.cuda.libdevice.llrint($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let realTerm ← realMathTerm "tl.extra.cuda.libdevice.llrint" e'
      let sh ← e'.shape.term
      let argCompute ←
        match e'.computeTerm with
        | some computeTerm => pure computeTerm
        | none => `(ComputeExpr.alg $realTerm)
      pure ⟨← `(Op.ref TileDType.int $sh "__compute_llrint_unprojected__"),
        .int, e'.shape, some (← `(ComputeExpr.llrint $argCompute)), some .int⟩
  | `(tritonExpr| -inf) =>
      pure ⟨← `(Op.negInf), .real, SInfo.scalar, none, none⟩
  | `(tritonExpr| float ($n:num)) => do
      unless n.getNat == 0 do
        Macro.throwError "float(integer): only the exact builtin literal float(0) is supported"
      expandExpr env (← `(tritonExpr| 0.0))
  | `(tritonExpr| - float ($arg:term)) => do
      let `(term| $value:str) := arg
        | Macro.throwError "float(...): expected a literal infinity string"
      if value.getString == "inf" then
        return ⟨← `(Op.negInf), .real, SInfo.scalar, none, none⟩
      if value.getString == "-inf" then
        let dtype := mkIdent `dtype
        return ← expandExpr env (← `(tritonExpr|
          tl.bitcast(tl.full([], 2139095040, $dtype:ident=tl.uint32), tl.float32)))
      Macro.throwError "float(...): only literal inf and -inf are supported"
  | `(tritonExpr| float ($arg:term)) => do
      let `(term| $value:str) := arg
        | Macro.throwError "float(...): expected a literal infinity string"
      if value.getString == "-inf" then
        return ⟨← `(Op.negInf), .real, SInfo.scalar, none, none⟩
      if value.getString == "inf" then
        -- Infinity has an exact binary32 encoding. Use the existing literal
        -- backend, with no finite Real proxy or numerical identity assumption.
        let dtype := mkIdent `dtype
        return ← expandExpr env (← `(tritonExpr|
          tl.bitcast(tl.full([], 2139095040, $dtype:ident=tl.uint32), tl.float32)))
      Macro.throwError "float(...): only literal inf and -inf are supported"
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr)) => do
      expandDot expandExpr env a b
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let (options, resultHint) ← expandDotOptions kwargs
      expandDot expandExpr env a b (some options) resultHint
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr)) => do
      expandDotAcc expandExpr env a b acc
  | `(tritonExpr| tl.dot($a:tritonExpr, $b:tritonExpr, $acc:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let (options, resultHint) ← expandDotOptions kwargs
      expandDotAcc expandExpr env a b acc (some options) resultHint
  | `(tritonExpr| tl.make_block_ptr($p:tritonExpr, $baseKw:ident=$base:tritonExpr,
        $shapeKw:ident=[$parentDims:tritonExpr,*], $stridesKw:ident=[$strideDims:tritonExpr,*],
        $offsetsKw:ident=[$offsetDims:tritonExpr,*], $blockShapeKw:ident=[$blockDims:tritonExpr,*])) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape` in that order"
      let sp ← expandStaticPtrExpr env p
      let (region, baseFromPtr?) ← match sp with
        | some sp =>
            pure (sp.region, if sp.baseOnly then none else some sp.offset)
        | none =>
            Macro.throwError "tl.make_block_ptr: first argument must be a region antiquote like `$(xReg)`"
      let base' ←
        match ← expandLeanAntiquoteAs? .nat base with
        | some out => pure out
        | none =>
            match base with
            | `(tritonExpr| $n:num) =>
                pure ⟨← `(Op.constNat $n), .nat, SInfo.scalar, none, none⟩
            | _ => expandNatExpectedExpr env base
      ensureShape SInfo.scalar base'.shape "tl.make_block_ptr base"
      let baseTerm ←
        match baseFromPtr? with
        | none => pure base'.term
        | some ptrBase => do
            let (bc, outShape) ← broadcastTerm SInfo.scalar SInfo.scalar "tl.make_block_ptr base"
            ensureShape SInfo.scalar outShape "tl.make_block_ptr base"
            `(Op.add NumericDType.nat $bc $ptrBase $base'.term)
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, offsetOps, hasDynamicOffsets) ←
        natBlockPtrOffsets env "tl.make_block_ptr" offsetDims
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      let term ←
        if hasDynamicOffsets then
          `(Op.makeBlockPtrDynOffsets $region $baseTerm $parentShape $blockShape $strides $offsetOps)
        else
          `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets)
      pure ⟨term, .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.make_block_ptr($baseKw:ident=$p:tritonExpr,
        $shapeKw:ident=($parentDims:tritonExpr,*), $stridesKw:ident=($strideDims:tritonExpr,*),
        $offsetsKw:ident=($offsetDims:tritonExpr,*), $blockShapeKw:ident=($blockDims:tritonExpr,*),
        $orderKw:ident=($_order:num,*))) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" &&
          orderKw.getId.toString == "order" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape`, optional `order` in that order"
      let sp ← expandStaticPtrExpr env p
      let (region, baseTerm, baseShape) ← match sp with
        | some sp => pure (sp.region, sp.offset, sp.shape)
        | none => Macro.throwError "tl.make_block_ptr: `base=` must start from a region expression"
      ensureShape SInfo.scalar baseShape "tl.make_block_ptr base"
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, offsetOps, hasDynamicOffsets) ←
        natBlockPtrOffsets env "tl.make_block_ptr" offsetDims
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      let term ←
        if hasDynamicOffsets then
          `(Op.makeBlockPtrDynOffsets $region $baseTerm $parentShape $blockShape $strides $offsetOps)
        else
          `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets)
      pure ⟨term, .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.make_block_ptr($baseKw:ident=$p:tritonExpr,
        $shapeKw:ident=($parentDims:tritonExpr,*), $stridesKw:ident=($strideDims:tritonExpr,*),
        $offsetsKw:ident=($offsetDims:tritonExpr,*), $blockShapeKw:ident=($blockDims:tritonExpr,*))) => do
      unless baseKw.getId.toString == "base" &&
          shapeKw.getId.toString == "shape" &&
          stridesKw.getId.toString == "strides" &&
          offsetsKw.getId.toString == "offsets" &&
          blockShapeKw.getId.toString == "block_shape" do
        Macro.throwError "tl.make_block_ptr kwargs must be `base`, `shape`, `strides`, `offsets`, `block_shape` in that order"
      let sp ← expandStaticPtrExpr env p
      let (region, baseTerm, baseShape) ← match sp with
        | some sp => pure (sp.region, sp.offset, sp.shape)
        | none => Macro.throwError "tl.make_block_ptr: `base=` must start from a region expression"
      ensureShape SInfo.scalar baseShape "tl.make_block_ptr base"
      let (parentShape, _) ← natListTerm "tl.make_block_ptr shape" parentDims
      let (strides, _) ← natListTerm "tl.make_block_ptr strides" strideDims
      let (offsets, offsetOps, hasDynamicOffsets) ←
        natBlockPtrOffsets env "tl.make_block_ptr" offsetDims
      let (blockShape, blockShapeInfo) ← natListTerm "tl.make_block_ptr block_shape" blockDims
      let term ←
        if hasDynamicOffsets then
          `(Op.makeBlockPtrDynOffsets $region $baseTerm $parentShape $blockShape $strides $offsetOps)
        else
          `(Op.makeBlockPtrDyn $region $baseTerm $parentShape $blockShape $strides $offsets)
      pure ⟨term, .blockPtr, .dims blockShapeInfo, none, none⟩
  | `(tritonExpr| tl.advance($p:tritonExpr, ($deltas:tritonExpr,*)))
  | `(tritonExpr| tl.advance($p:tritonExpr, [$deltas:tritonExpr,*])) => do
      let p' ← expandExpr env p
      ensureDType .blockPtr p'.dtype "tl.advance pointer"
      let (constants, ops, dynamic) ← blockPtrDeltas env deltas
      let term ← if dynamic then `(Op.advanceBlockPtrDyn $p'.term $ops)
        else `(Op.advanceBlockPtr $p'.term $constants)
      pure ⟨term, .blockPtr, p'.shape, none, none⟩
  | `(tritonExpr| tl.advance($p:tritonExpr, $offsetsKw:ident=($deltas:tritonExpr,*))) => do
      unless offsetsKw.getId.toString == "offsets" do
        Macro.throwError "tl.advance kwarg must be `offsets`"
      let p' ← expandExpr env p
      ensureDType .blockPtr p'.dtype "tl.advance pointer"
      let (constants, ops, dynamic) ← blockPtrDeltas env deltas
      let term ← if dynamic then `(Op.advanceBlockPtrDyn $p'.term $ops)
        else `(Op.advanceBlockPtr $p'.term $constants)
      pure ⟨term, .blockPtr, p'.shape, none, none⟩
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs (positionalMask := some mask)
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      let dst ← expandDType dt
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := some dst) (positionalMask := some mask)
      if let some cast ← machineIntegerCast? dt loaded then return cast
      return loaded
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_dt:ident)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs (positionalMask := some mask)
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_base:ident.$_field:ident)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs (positionalMask := some mask)
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => do
      let dst ← expandDType dt
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := some dst)
      if let some cast ← machineIntegerCast? dt loaded then return cast
      return loaded
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_dt:ident)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_base:ident.$_field:ident)) => do
      expandLoad expandExpr expandStaticPtrExpr env p kwargs
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.lt)) a b
  | `(tritonExpr| $a:tritonExpr <= $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.le)) a b
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.eq)) a b
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.gt)) a b
  | `(tritonExpr| $a:tritonExpr >= $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.ge)) a b
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
      expandCmp expandExpr env "comparison" (← `(Op.ne)) a b
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => do
      expandBoolOrNatBitwise expandExpr env "`&`" (← `(Op.boolAnd)) (← `(Op.bitAnd)) a b
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => do
      expandBoolBin expandExpr env "`and`" (← `(Op.boolAnd)) a b
  | `(tritonExpr| $a:tritonExpr | $b:tritonExpr) => do
      expandBoolOrNatBitwise expandExpr env "`|`" (← `(Op.boolOr)) (← `(Op.bitOr)) a b
  | `(tritonExpr| $a:tritonExpr ^ $b:tritonExpr) => do
      expandNatBitwise expandExpr env "`^`" (← `(Op.bitXor)) a b
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) => do
      let left ← expandExpr env a
      -- Retained native widths use the fixed-width path below. Only the
      -- explicitly mathematical Int carrier keeps an unbounded shift.
      if left.dtype == .int then
        let right ← expandNatExpectedExpr env b
        let (bc, shape) ← broadcastTerm left.shape right.shape "signed left shift"
        return ⟨← `(Op.intShiftLeft $bc $left.term $right.term), .int, shape, none, none⟩
      expandNatBitwise expandExpr env "`<<`" (← `(Op.shiftLeft)) a b
  | `(tritonExpr| $a:tritonExpr >> $b:tritonExpr) => do
      expandNatBitwise expandExpr env "`>>`" (← `(Op.shiftRight)) a b
  | `(tritonExpr| ~$a:tritonExpr) => do
      expandBoolNot expandExpr env "boolean ~" a
  | `(tritonExpr| not $a:tritonExpr) => do
      expandBoolNot expandExpr env "boolean not" a
  | `(tritonExpr| -$a:tritonExpr) => do
      -- Literal negation stays a constant; value negation is a sign-bit op.
      match a with
      | `(tritonExpr| $n:num) =>
          return ⟨← `(Op.const (-$n)), .real, SInfo.scalar, none, none⟩
      | `(tritonExpr| $n:scientific) =>
          return ⟨← `(Op.const (-$n)), .real, SInfo.scalar, none, none⟩
      | _ => pure ()
      let a' := (← expandExpr env a).forgetStorageWidth
      match a'.dtype with
      | .int =>
          let (bc, outShape) ← broadcastTerm SInfo.scalar a'.shape "unary -"
          pure ⟨← `(Op.sub NumericDType.int $bc (Op.constInt 0) $a'.term),
            .int, outShape, none, none⟩
      | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ =>
          let dtype ← a'.dtype.floatProof
          let term ← `(Op.negFloat $dtype $a'.term)
          match a'.computeDType? with
          | some .fp32 =>
              let child ← a'.scopedFloatingTerm
              let fp ← `(Op.negFloat $dtype $child)
              pure ⟨term, a'.dtype, a'.shape, some (← fp32ComputeExpr fp), some .fp32⟩
          | some .real =>
              let child ← a'.scopedFloatingTerm
              let fp ← `(Op.negFloat $dtype $child)
              pure ⟨term, a'.dtype, a'.shape, some (← fp64ComputeExpr fp), some .real⟩
          | _ =>
              if a'.computeTerm.isSome then
                let child ← a'.floatingTerm
                let fp ← `(Op.negFloat $dtype $child)
                pure ⟨term, a'.dtype, a'.shape, some (← `(ComputeExpr.alg $fp)), none⟩
              else pure ⟨term, a'.dtype, a'.shape, none, none⟩
      | _ => Macro.throwError "unary -: dtype mismatch"
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      match ← expandStaticPtrExpr env stx with
      | some sp =>
          let (bc, _) ← broadcastTerm SInfo.scalar sp.shape "pointer arithmetic"
          pure ⟨← `(Op.ptrAdd $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape, none, none⟩
      | none =>
          let a' := (← expandExpr env a).forgetStorageWidth
          let b' := (← expandExpr env b).forgetStorageWidth
          match a'.dtype, b'.dtype with
          | .ptr, .nat =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $a'.term $b'.term), .ptr, outShape, none, none⟩
          | .ptr, .int =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer arithmetic"
              let bNat ← pointerOffsetIntAsNat b'
              pure ⟨← `(Op.ptrAdd $bc $a'.term $bNat.term), .ptr, outShape, none, none⟩
          | .nat, .ptr =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm b'.shape a'.shape "pointer arithmetic"
              pure ⟨← `(Op.ptrAdd $bc $b'.term $a'.term), .ptr, outShape, none, none⟩
          | .int, .ptr =>
              ensureAlgorithmOnly "pointer arithmetic" a'
              ensureAlgorithmOnly "pointer arithmetic" b'
              let (bc, outShape) ← broadcastTerm b'.shape a'.shape "pointer arithmetic"
              let aNat ← pointerOffsetIntAsNat a'
              pure ⟨← `(Op.ptrAdd $bc $b'.term $aNat.term), .ptr, outShape, none, none⟩
          | _, _ =>
              expandArith expandExpr env "arithmetic" (← `(Op.add)) a b
  | `(tritonExpr| $a:tritonExpr - float ($arg:term)) => do
      let argString := toString arg.raw
      unless argString.contains "inf" && !argString.contains "-inf" do
        Macro.throwError "float(...): subtraction shorthand only supports `float(\"inf\")`"
      expandExpr env (← `(tritonExpr| $a:tritonExpr + -inf))
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => do
      let a' ← expandExpr env a
      if a'.dtype != .ptr then
        expandArith expandExpr env "arithmetic" (← `(Op.sub)) a b
      else
        match ← expandStaticPtrExpr env stx with
        | some sp =>
            let (bc, _) ← broadcastTerm SInfo.scalar sp.shape "pointer subtraction"
            pure ⟨← `(Op.ptrAdd $bc (Op.ptrBase $sp.region) $sp.offset), .ptr, sp.shape, none, none⟩
        | none =>
            let b' ← expandNatExpectedExpr env b
            ensureAlgorithmOnly "pointer subtraction" a'
            ensureAlgorithmOnly "pointer subtraction" b'
            let (bc, outShape) ← broadcastTerm a'.shape b'.shape "pointer subtraction"
            pure ⟨← `(Op.ptrSub $bc $a'.term $b'.term), .ptr, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      match ← expandBoolMaskMul? expandExpr env a b with
      | some out => pure out
      | none => expandArith expandExpr env "arithmetic" (← `(Op.mul)) a b
  | `(tritonExpr| $a:tritonExpr / $b:tritonExpr) => do
      expandArith expandExpr env "arithmetic" (← `(Op.div)) a b
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      expandIntegralArith expandExpr env "runtime integer division" (← `(Op.tritonDiv)) a b
  | `(tritonExpr| _div_up($a:tritonExpr, $b:tritonExpr)) => do
      let one ← `(Op.constNat 1)
      let a' ← expandNatExpectedExpr env a
      let b' ← expandNatExpectedExpr env b
      let (bc, outShape) ← broadcastTerm a'.shape b'.shape "_div_up"
      let num ← `(Op.sub NumericDType.nat $bc
        (Op.add NumericDType.nat $bc $a'.term $b'.term) $one)
      pure ⟨← `(Op.floorDiv IntegralDType.nat $bc $num $b'.term),
        .nat, outShape, none, none⟩
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      expandIntegralArith expandExpr env "runtime integer remainder" (← `(Op.tritonMod)) a b
  | `(tritonExpr| tl.where($c:tritonExpr, $a:tritonExpr, $b:tritonExpr)) => do
      let c' ← expandExpr env c
      expandWhereFromCond env c' a b
  | `(tritonExpr| ($a:tritonExpr if $c:tritonExpr else $b:tritonExpr)) => do
      let c' ← expandBoolCondition env c
      expandIteFromCond env c' a b
  | `(tritonExpr| $e:tritonExpr[ : ]) => do
      -- Python no-op full slice `e[:]`.
      expandExpr env e
  | `(tritonExpr| $e:tritonExpr[ : , None ]) => do
      -- `e[:, None]` — insert a unit axis at position 1: `[N] → [N, 1]`.
      expandSlicerNone expandExpr env e (axisIdx := 1)
  | `(tritonExpr| $e:tritonExpr[ None , : ]) => do
      -- `e[None, :]` — insert a unit axis at position 0: `[N] → [1, N]`.
      expandSlicerNone expandExpr env e (axisIdx := 0)
  | `(tritonExpr| $e:tritonExpr[ None ]) => do
      -- `e[None]` — insert a unit axis at position 0: `[] → [1]`.
      expandExpandDims expandExpr env e (axisIdx := 0)
  | `(tritonExpr| $e:tritonExpr[ : , None , None ]) => do
      -- `e[:, None, None]` — `[N] → [N, 1, 1]`.
      let e1 ← expandExpandDims expandExpr env e (axisIdx := 1)
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit "2"⟩
      let dims := match e1.shape with | .dims ds => ds
      let outDims : List (TSyntax `term) :=
        dims.take 2 ++ [← `((1 : Nat))] ++ dims.drop 2
      pure ⟨← `(Op.expandDim (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.setAxisOne]⟩) $e1.term),
        e1.dtype, .dims outDims, e1.computeTerm, e1.computeDType?⟩
  | `(tritonExpr| $e:tritonExpr[ None , : , None ]) => do
      let e1 ← expandExpandDims expandExpr env e (axisIdx := 0)
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit "2"⟩
      let dims := match e1.shape with | .dims ds => ds
      let outDims : List (TSyntax `term) :=
        dims.take 2 ++ [← `((1 : Nat))] ++ dims.drop 2
      pure ⟨← `(Op.expandDim (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.setAxisOne]⟩) $e1.term),
        e1.dtype, .dims outDims, e1.computeTerm, e1.computeDType?⟩
  | `(tritonExpr| $e:tritonExpr[ None , None , : ]) => do
      let e1 ← expandExpandDims expandExpr env e (axisIdx := 0)
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit "0"⟩
      let dims := match e1.shape with | .dims ds => ds
      let outDims : List (TSyntax `term) :=
        [← `((1 : Nat))] ++ dims
      pure ⟨← `(Op.expandDim (⟨$axisLit, by simp [TileShape.eraseAxis, TileShape.setAxisOne]⟩) $e1.term),
        e1.dtype, .dims outDims, e1.computeTerm, e1.computeDType?⟩
  | `(tritonExpr| $e:tritonExpr[ : , None , : ]) => do
      -- `e[:, None, :]` — insert a unit axis in the middle of a rank-2 tile.
      expandExpandDims expandExpr env e (axisIdx := 1)
  | `(tritonExpr| $e:tritonExpr[ None , : , : ]) => do
      -- `e[None, :, :]` — add a leading unit axis to a rank-2 tile.
      expandExpandDims expandExpr env e (axisIdx := 0)
  | `(tritonExpr| $e:tritonExpr[ : , : , None ]) => do
      -- `e[:, :, None]` — add a trailing unit axis to a rank-2 tile.
      expandExpandDims expandExpr env e (axisIdx := 2)
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $kw:tritonReduceKwarg)) => do
      match kw with
      | `(tritonReduceKwarg| axis = $n:num) =>
          expandExpandDims expandExpr env e (axisIdx := n.getNat)
      | `(tritonReduceKwarg| $name:ident = $_) =>
          Macro.throwError
            ("tl.expand_dims: unknown kwarg `" ++ name.getId.toString ++
             "`. Only literal `axis = N` is supported.")
      | _ => Macro.throwUnsupported
  | `(tritonExpr| tl.expand_dims($e:tritonExpr, $n:num)) => do
      expandExpandDims expandExpr env e (axisIdx := n.getNat)
  | `(tritonExpr| tl.trans($e:tritonExpr)) => do
      -- `tl.trans(e)` — transpose the trailing two axes (`Op.transpose`).
      expandTranspose expandExpr env e
  | `(tritonExpr| tl.permute($e:tritonExpr, [$axes:num,*])) => do
      expandPermute expandExpr env e axes.getElems.toList
  | `(tritonExpr| tl.reshape($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike expandExpr env "tl.reshape" e dims.getElems
  | `(tritonExpr| tl.broadcast_to($e:tritonExpr, [$dims:tritonExpr,*])) => do
      -- `tl.broadcast_to(e, [dims*])`: broadcast `e` to the literal target
      -- shape, reusing the mutual-broadcast machinery (`Op.broadcast` for a
      -- scalar source, `Op.remap` over `Broadcast.leftIndex` otherwise; an
      -- already-matching shape is the identity).
      let e' ← expandExpr env e
      let (_, targetShape) ← expandShapeDims "tl.broadcast_to" dims.getElems
      let out ← coerceShape e'.term e'.shape targetShape "tl.broadcast_to"
      pure ⟨out, e'.dtype, targetShape, none, none⟩
  | `(tritonExpr| tl.view($e:tritonExpr, [$dims:tritonExpr,*])) => do
      expandReshapeLike expandExpr env "tl.view" e dims.getElems
  | `(tritonExpr| tl.ravel($e:tritonExpr)) => do
      expandRavel expandExpr env e
  | `(tritonExpr| tl.flip($e:tritonExpr, $ax:num)) => do
      expandFlip expandExpr env e ax.getNat
  | `(tritonExpr| tl.flip($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let ax ← parseFlipKwargs kwargs
      expandFlip expandExpr env e ax
  | `(tritonExpr| tl.join($a:tritonExpr, $b:tritonExpr)) => do
      expandJoin expandExpr env a b
  | `(tritonExpr| tl.interleave($a:tritonExpr, $b:tritonExpr)) => do
      expandInterleave expandExpr env a b
  | `(tritonExpr| tl.split($e:tritonExpr, $side:num)) => do
      expandSplit expandExpr env e side.getNat
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr)) => do
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr, $dt:tritonDType)) => do
      expandComputeFull expandExpr env dims.getElems v dt
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr, $name:ident=$dt:tritonDType)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError
          ("tl.full: unknown kwarg `" ++ name.getId.toString ++ "`. Only `dtype=` is recognized.")
      expandComputeFull expandExpr env dims.getElems v dt
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr,
      $name:ident=$ptr:ident.$dtype:ident.$elem:ident)) => do
      unless name.getId.getString! == "dtype" &&
          dtype.getId.getString! == "dtype" && elem.getId.getString! == "element_ty" do
        Macro.throwError "tl.full: expected `dtype=<ptr>.dtype.element_ty`"
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $v:tritonExpr, $name:ident=$dt:ident)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError "tl.full: expected `dtype=`"
      let dtTerm : TSyntax `term := ⟨dt.raw⟩
      expandFullDTypeTerm expandExpr env dims.getElems v dtTerm
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $dtypeName:ident=$dt:tritonDType,
      $valueName:ident=$v:tritonExpr)) => do
      unless dtypeName.getId.getString! == "dtype" && valueName.getId.getString! == "value" do
        Macro.throwError "tl.full: expected kwargs `dtype=` and `value=`"
      expandComputeFull expandExpr env dims.getElems v dt
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $valueName:ident=$v:tritonExpr,
      $dtypeName:ident=$dt:tritonDType)) => do
      unless valueName.getId.getString! == "value" && dtypeName.getId.getString! == "dtype" do
        Macro.throwError "tl.full: expected kwargs `value=` and `dtype=`"
      expandComputeFull expandExpr env dims.getElems v dt
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $dtypeName:ident=$ptr:ident.$dtype:ident.$elem:ident,
      $valueName:ident=$v:tritonExpr)) => do
      unless dtypeName.getId.getString! == "dtype" &&
          valueName.getId.getString! == "value" &&
          dtype.getId.getString! == "dtype" && elem.getId.getString! == "element_ty" do
        Macro.throwError "tl.full: expected kwargs `dtype=<ptr>.dtype.element_ty` and `value=`"
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $valueName:ident=$v:tritonExpr,
      $dtypeName:ident=$ptr:ident.$dtype:ident.$elem:ident)) => do
      unless valueName.getId.getString! == "value" &&
          dtypeName.getId.getString! == "dtype" &&
          dtype.getId.getString! == "dtype" && elem.getId.getString! == "element_ty" do
        Macro.throwError "tl.full: expected kwargs `value=` and `dtype=<ptr>.dtype.element_ty`"
      expandFull expandExpr env dims.getElems v
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $dtypeName:ident=$dt:ident,
      $valueName:ident=$v:tritonExpr)) => do
      unless dtypeName.getId.getString! == "dtype" &&
          valueName.getId.getString! == "value" do
        Macro.throwError "tl.full: expected kwargs `dtype=` and `value=`"
      let dtTerm : TSyntax `term := ⟨dt.raw⟩
      expandFullDTypeTerm expandExpr env dims.getElems v dtTerm
  | `(tritonExpr| tl.full([$dims:tritonExpr,*], $valueName:ident=$v:tritonExpr,
      $dtypeName:ident=$dt:ident)) => do
      unless valueName.getId.getString! == "value" &&
          dtypeName.getId.getString! == "dtype" do
        Macro.throwError "tl.full: expected kwargs `value=` and `dtype=`"
      let dtTerm : TSyntax `term := ⟨dt.raw⟩
      expandFullDTypeTerm expandExpr env dims.getElems v dtTerm
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*])) => do
      -- `tl.zeros([dims])` ≡ `tl.full([dims], 0)`.
      let zero ← `(tritonExpr| 0)
      expandFull expandExpr env dims.getElems zero
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $dt:tritonDType)) => do
      expandComputeZeros expandExpr env dims.getElems dt
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $name:ident=$dt:tritonDType)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError
          ("tl.zeros: unknown kwarg `" ++ name.getId.toString ++ "`. Only `dtype=` is recognized.")
      expandComputeZeros expandExpr env dims.getElems dt
  | `(tritonExpr| tl.zeros([$dims:tritonExpr,*], $name:ident=$dt:term)) => do
      unless name.getId.getString! == "dtype" do
        Macro.throwError
          ("tl.zeros: unknown kwarg `" ++ name.getId.toString ++ "`. Only `dtype=` is recognized.")
      expandZerosDTypeTerm expandExpr env dims.getElems dt
  | `(tritonExpr| tl.zeros_like($e:tritonExpr)) => do
      let e' ← expandExpr env e
      let zero ←
        match e'.dtype with
        | .fp32Argument =>
            Macro.throwError "tl.zeros_like: external FP32 argument must be read first"
        | .real => `(Op.const 0.0)
        | .fp32 => `(Op.constFloat FloatDType.fp32 0.0)
        | .fp16 => `(Op.constFloat FloatDType.fp16 0.0)
        | .bf16 => `(Op.constFloat FloatDType.bf16 0.0)
        | .f8e4 => `(Op.constFloat FloatDType.f8e4 0.0)
        | .f8e5 => `(Op.constFloat FloatDType.f8e5 0.0)
        | .floatVar name => do
            let t : TSyntax `term := ⟨mkIdent (Name.mkSimple name)⟩
            `(Op.constFloat $t 0.0)
        | .nat => `(Op.constNat 0)
        | .int | .int8 | .int32 => `(Op.constInt 0)
        | .bool => `(Op.constBool Bool.false)
        | .ptr | .blockPtr =>
            Macro.throwError "tl.zeros_like: pointer-valued tensors are not supported"
      let shape ← e'.shape.term
      pure ⟨← `(Op.full $shape $zero), e'.dtype, e'.shape, none, none⟩
  | _ => (Macro.throwUnsupported : MacroM EOut)

end

/-- Explicit value conversions can establish a new type for a reused source
name, including when the conversion is nested inside an arithmetic expression.
The caller still checks the expanded result type before overriding a pin. -/
private partial def containsExplicitConversion (env : Env) (raw : Syntax) : Bool :=
  let e : TSyntax `tritonExpr := ⟨raw⟩
  (methodCast? e).isSome || (methodCastRegisterDType? env e).isSome ||
    (match e with
      | `(tritonExpr| tl.toReal($_:tritonExpr)) => Bool.true
      | `(tritonExpr| tl.cast($_:tritonExpr, $_:tritonDType)) => Bool.true
      | _ => Bool.false) || raw.getArgs.any (containsExplicitConversion env)

/-- Carry known floating and int32 storage into nested load expressions, and
preserve integer storage underneath unsigned casts. This is
internal elaboration metadata: the surface source and its operations stay the
same. Unknown roots and explicit load dtypes are left unchanged. -/
partial def annotateFloatingLoadStorage (regions : Inference.RegionDTypes)
    (pointers : Inference.PtrElems) (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args => do
        let args ← args.mapM (annotateFloatingLoadStorage regions pointers)
        pure (.node info kind args)
    | other => pure other
  let keyword? (p : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg)
      (unsignedCast : Bool := Bool.false) :
      MacroM (Option (TSyntax `tritonMemKwarg)) := do
    for kw in kwargs do
      match kw with
      | `(tritonMemKwarg| $name:ident = $_:tritonDType)
      | `(tritonMemKwarg| $name:ident = $_:tritonExpr) =>
          if name.getId.eraseMacroScopes.toString == "dtype" then return none
      | _ => pure ()
    let dtype ← match Inference.pointerElementDType? regions pointers p with
      | some .fp16 => `(tritonDType| tl.float16)
      | some .bf16 => `(tritonDType| tl.bfloat16)
      | some .f8e4 => `(tritonDType| tl.float8e4nv)
      | some .f8e5 => `(tritonDType| tl.float8e5)
      | some .int32 => `(tritonDType| tl.int32)
      | some .int =>
          if unsignedCast then `(tritonDType| tl.int64) else return none
      | some .int8 =>
          if unsignedCast then `(tritonDType| tl.int8) else return none
      | some .nat =>
          if unsignedCast then `(tritonDType| tl.uint64) else return none
      | some (.floatVar "dtype") => `(tritonDType| DTYPE)
      | some (.floatVar "out_dtype") => `(tritonDType| OUT_DTYPE)
      | _ => return none
    let kwName : Ident := mkIdent `dtype
    return some (← `(tritonMemKwarg| $kwName:ident = $dtype:tritonDType))
  let expr : TSyntax `tritonExpr := ⟨raw⟩
  let hasExplicitDType (kwargs : TSyntaxArray `tritonMemKwarg) : Bool :=
    kwargs.any fun kw => match kw with
      | `(tritonMemKwarg| $name:ident = $_:tritonDType)
      | `(tritonMemKwarg| $name:ident = $_:tritonExpr) =>
          name.getId.eraseMacroScopes.toString == "dtype"
      | _ => Bool.false
  -- Machine integer casts may contain arbitrarily parenthesized loads. Preserve the
  -- load's declared channel and canonicalize the cast even if a dtype kwarg
  -- is already present; that kwarg must not cause the conversion to disappear.
  let rec unsignedLoad? (e : TSyntax `tritonExpr) :
      MacroM (Option (TSyntax `tritonExpr)) := do
    match e with
    | `(tritonExpr| ($inner:tritonExpr)) => unsignedLoad? inner
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
        let kwargs ← match ← keyword? p kwargs Bool.true with
          | some kw => pure (kwargs.push kw)
          | none =>
              if hasExplicitDType kwargs then pure kwargs else return none
        return some (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
        let kwargs ← match ← keyword? p kwargs Bool.true with
          | some kw => pure (kwargs.push kw)
          | none =>
              if hasExplicitDType kwargs then pure kwargs else return none
        return some (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)))
    | _ => return none
  let cast? ← match expr with
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
        pure (some (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)), dt))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
        pure (some (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)), dt))
    | `(tritonExpr| tl.cast($e:tritonExpr, $dt:tritonDType)) => pure (some (e, dt))
    | _ => pure (methodCast? expr)
  if let some (operand, dtype) := cast? then
    if (unsignedCastWidth? dtype).isSome || signedMachineCast dtype then
      if let some loaded ← unsignedLoad? operand then
        return (← `(tritonExpr| ($loaded:tritonExpr).to($dtype:tritonDType))).raw
  match expr with
  | `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
      let some kw ← keyword? p kwargs ((unsignedCastWidth? dt).isSome || signedMachineCast dt) | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
      let some kw ← keyword? p kwargs ((unsignedCastWidth? dt).isSome || signedMachineCast dt) | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  -- Direct-load attribute casts have their own parser nodes. Canonicalize
  -- them only when source storage is known, so the regular attribute-cast
  -- resolver sees both the original load channel and the register's dtype.
  -- Merely erasing `.to(other.dtype)` would lose two numerical events.
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($target:ident))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($target:ident))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($base:ident.$field:ident)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($base:ident.$field:ident))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($base:ident.$field:ident)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($base:ident.$field:ident))).raw
  | `(tritonExpr| tl.cast(tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*), $dt:tritonDType)) =>
      if (unsignedCastWidth? dt).isNone && !signedMachineCast dt then return raw
      let some kw ← keyword? p kwargs Bool.true | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.cast(tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*), $dt:tritonDType))).raw
  | `(tritonExpr| tl.cast(tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*), $dt:tritonDType)) =>
      if (unsignedCastWidth? dt).isNone && !signedMachineCast dt then return raw
      let some kw ← keyword? p kwargs Bool.true | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.cast(tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*), $dt:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | _ => return raw

/-- Floating storage evidence is separate from ordinary dtype inference.
An unknown binding shadows older evidence, including a same-named region. -/
abbrev FPPtrProvenance := List (String × Option DInfo)

/-- Storage-profile presence is a property of the complete source, not of the
current pointer map. Only unprofiled mathematical surfaces retain their legacy
pointer-element cast interpretation. -/
structure FPSourceContext where
  regions : Inference.RegionDTypes
  legacyPointerCasts : Bool


/-- Normalize only the syntax inspected for floating provenance. A mathematical
`Region.real` annotation supplies no fp64 evidence; a named root must obtain
that evidence from an explicit region declaration. Emitted source is unchanged. -/
private partial def fpProvenanceSyntax (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args =>
        pure (.node info kind (← args.mapM fpProvenanceSyntax))
    | other => pure other
  let term : TSyntax `term := ⟨raw⟩
  match term with
  | `(($inner:term : Region .real))
  | `(($inner:term : Region TileDType.real)) => return inner.raw
  | _ => pure ()
  let expr : TSyntax `tritonExpr := ⟨raw⟩
  match expr with
  | `(tritonExpr| $($name:ident)) => return (← `(tritonExpr| $name:ident)).raw
  | _ => return raw

private def fpPointerDType? (regions : Inference.RegionDTypes)
    (pointers : FPPtrProvenance) (p : TSyntax `tritonExpr) : MacroM (Option DInfo) := do
  let regions := regions.filter fun entry =>
    !(pointers.any fun binding => binding.1 == entry.1)
  let known := pointers.filterMap fun (name, dtype) => dtype.map (name, ·)
  Inference.pointerElementDType? regions known ⟨← fpProvenanceSyntax p.raw⟩ |> pure

/-- Names assigned under control flow lose evidence after that statement. This
conservative join never infers fp64 from just one branch or a possibly empty loop. -/
private partial def fpAssignedNames (raw : Syntax) : List String :=
  let st : TSyntax `tritonStmt := ⟨raw⟩
  match st with
  | `(tritonStmt| $name:ident = $_:tritonExpr)
  | `(tritonStmt| $name:ident := $_:tritonExpr) => [name.getId.toString]
  | _ => raw.getArgs.toList.flatMap fpAssignedNames

/-- Resolve a value's dtype attribute from its retained source precision.
In particular an FP32 value uses the Real carrier plus an FP32 annotation;
looking at the carrier alone would erase its explicit load conversion. -/
private partial def annotateValueDTypeCasts (env : Env) (legacy : Bool)
    (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args => pure (.node info kind (← args.mapM (annotateValueDTypeCasts env legacy)))
    | other => pure other
  let e : TSyntax `tritonExpr := ⟨raw⟩
  let some (.str valueName "dtype") := methodCastTargetName? e | return raw
  if valueName.isAnonymous then return raw
  let entry := env.find? (fun entry => entry.1 == valueName.toString || entry.1 == valueName.getString!)
  let precision : Option (DInfo × Option DInfo) := entry.map (fun e => (e.2.1, e.2.2.2))
  let dtype : Option (TSyntax `tritonDType) ← match precision with
    | some (.real, some .fp32) | some (.fp32, _) => pure (some (← `(tritonDType| tl.float32)))
    | some (.real, some .real) => pure (some (← `(tritonDType| tl.float64)))
    | some (.fp16, _) => pure (some (← `(tritonDType| tl.float16)))
    | some (.bf16, _) => pure (some (← `(tritonDType| tl.bfloat16)))
    | some (.f8e4, _) => pure (some (← `(tritonDType| tl.float8e4nv)))
    | some (.f8e5, _) => pure (some (← `(tritonDType| tl.float8e5)))
    | some (.floatVar "dtype", _) => pure (some (← `(tritonDType| DTYPE)))
    | some (.floatVar "out_dtype", _) => pure (some (← `(tritonDType| OUT_DTYPE)))
    | _ => pure none
  let some dtype := dtype | do
    if !legacy then Macro.throwError "value .dtype cast requires known source precision"
    return raw
  match e with
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($_:ident)) =>
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType))).raw
  | _ =>
      if raw.getNumArgs > 0 then
        let value : TSyntax `tritonExpr := ⟨raw[0]⟩
        return (← `(tritonExpr| ($value:tritonExpr).to($dtype:tritonDType))).raw
      return raw

/-- Resolve a pointer element cast from storage evidence at this program point.
This changes only internal syntax; explicit source `.to(p.dtype.element_ty)`
still contributes its storage conversion to the compute program. -/
private partial def annotatePointerElementCasts (regions : Inference.RegionDTypes)
    (legacyPointerCasts : Bool) (pointers : FPPtrProvenance) (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args =>
        pure (.node info kind (← args.mapM (annotatePointerElementCasts regions legacyPointerCasts pointers)))
    | other => pure other
  unless raw.getKind == ``tritonMethodCastElementTy ||
      raw.getKind == ``tritonMethodCastDTypeIdent ||
      raw.getKind == ``tritonMethodCastElementTyIdent ||
      raw.getKind == ``tritonIdentMethodCastDTypeIdent ||
      raw.getKind == ``tritonIdentMethodCastDTypeIdentSpaced ||
      raw.getKind == ``tritonIdentMethodCastElementTyIdent ||
      raw.getKind == ``tritonIdentMethodCastElementTyIdentSpaced do
    return raw
  let expr : TSyntax `tritonExpr := ⟨raw⟩
  let some (.str (.str pointer field) "element_ty") := methodCastTargetName? expr
    | return raw
  unless field == "dtype" || field == "type" do return raw
  let p : Ident := mkIdent pointer
  let dtype ← match ← fpPointerDType? regions pointers (← `(tritonExpr| $p:ident)) with
    | some .real => `(tritonDType| tl.float64)
    | some .fp32 => `(tritonDType| tl.float32)
    | some .fp16 => `(tritonDType| tl.float16)
    | some .bf16 => `(tritonDType| tl.bfloat16)
    | some .f8e4 => `(tritonDType| tl.float8e4nv)
    | some .f8e5 => `(tritonDType| tl.float8e5)
    | some (.floatVar "dtype") => `(tritonDType| DTYPE)
    | some (.floatVar "out_dtype") => `(tritonDType| OUT_DTYPE)
    | some .bool => `(tritonDType| tl.int1)
    | _ =>
        if field == "type" && !legacyPointerCasts then
          Macro.throwError "pointer .type.element_ty cast requires known supported storage provenance"
        return raw
  let args := raw.getArgs
  if h : 0 < args.size then
    let value : TSyntax `tritonExpr := ⟨args[0]⟩
    return (← `(tritonExpr| ($value:tritonExpr).to($dtype:tritonDType))).raw
  return raw

/-- A loop-carried binding cannot use evidence from only its first iteration.
Assignments earlier in the loop body may establish fresh evidence normally. -/
private def fpLoopProvenance (regions : Inference.RegionDTypes) (pointers : FPPtrProvenance) (index : Ident)
    (stmts : List (TSyntax `tritonStmt)) : FPPtrProvenance :=
  let assigned := index.getId.toString :: stmts.flatMap (fun st => fpAssignedNames st.raw)
  -- A fresh local is still unbound on the zero-iteration/untaken path.
  -- Only existing bindings and shadowed source regions need unknown markers.
  let shadowed := assigned.filter (fun name =>
    pointers.any (fun p => p.1 == name) || (Inference.lookupRegionDType regions name).isSome)
  shadowed.map (·, none) ++ pointers.filter (fun p => !(assigned.contains p.1))

/-- Definite evidence drives precision. Possible evidence is retained only
inside a control-flow analysis to reject conflicting typed-pointer bindings
even when one nested arm leaves the register unbound. -/
private abbrev FPPtrFlow := FPPtrProvenance × FPPtrProvenance

private def fpFlowLookup (regions : Inference.RegionDTypes)
    (pointers : FPPtrProvenance) (name : String) : Option (Option DInfo) :=
  match pointers.find? (fun p => p.1 == name) with
  | some p => some p.2
  | none => (Inference.lookupRegionDType regions name).map some

private def fpJoinFlows (regions : Inference.RegionDTypes)
    (base left right : FPPtrFlow) (assigned : List String) : MacroM FPPtrFlow := do
  let mut definite := base.1.filter (fun p => !(assigned.contains p.1))
  let mut possible := base.2.filter (fun p => !(assigned.contains p.1))
  for name in assigned do
    let ld := (fpFlowLookup regions left.1 name).join
    let rd := (fpFlowLookup regions right.1 name).join
    definite := (name, if ld == rd then ld else none) :: definite
    let lp := fpFlowLookup regions left.2 name
    let rp := fpFlowLookup regions right.2 name
    let choices ← match lp, rp with
      | some (some a), some (some b) =>
          if a == b then pure (some (some a))
          else Macro.throwError ("pointer storage dtype conflict across branches: " ++ name)
      | some (some _), some none | some none, some (some _) =>
          Macro.throwError ("pointer storage dtype is unknown on a bound branch: " ++ name)
      | none, other | other, none => pure other
      | some none, some none => pure (some none)
    -- An unbound fp64 arm cannot silently become a default-fp32 real load.
    -- Narrow channels still carry their dtype in ordinary inference, and a
    -- missing pointer register still fails dynamically. Wide partial joins
    -- require a future path-sensitive continuation and are rejected here.
    if choices == some (some DInfo.real) && (ld.isNone || rd.isNone) then
      Macro.throwError ("fp64 pointer storage is not established on every branch: " ++ name)
    if let some dtype := choices then
      possible := (name, dtype) :: possible
  return (definite, possible)

mutual

private partial def fpFlowAfterStmts (regions : Inference.RegionDTypes)
    (flow : FPPtrFlow) (stmts : List (TSyntax `tritonStmt)) : MacroM FPPtrFlow :=
  stmts.foldlM (fpFlowAfterStmt regions) flow

private partial def fpFlowAfterStmt (regions : Inference.RegionDTypes)
    (flow : FPPtrFlow) (st : TSyntax `tritonStmt) : MacroM FPPtrFlow := do
  match st with
  | `(tritonStmt| $name:ident = $e:tritonExpr)
  | `(tritonStmt| $name:ident := $e:tritonExpr) =>
      let dtype ← fpPointerDType? regions flow.1 e
      let possible ← fpPointerDType? regions flow.2 e
      let key := name.getId.toString
      return ((key, dtype) :: flow.1.filter (fun p => p.1 != key),
        (key, possible) :: flow.2.filter (fun p => p.1 != key))
  | `(tritonStmt| if $_:tritonExpr { $ts:tritonStmt* } else { $es:tritonStmt* })
  | `(tritonStmt| tl.if $_:tritonExpr { $ts:tritonStmt* } else { $es:tritonStmt* }) =>
      let left ← fpFlowAfterStmts regions flow ts.toList
      let right ← fpFlowAfterStmts regions flow es.toList
      fpJoinFlows regions flow left right (fpAssignedNames st.raw)
  | `(tritonStmt| if $_:tritonExpr { $ts:tritonStmt* })
  | `(tritonStmt| tl.if $_:tritonExpr { $ts:tritonStmt* }) =>
      let left ← fpFlowAfterStmts regions flow ts.toList
      fpJoinFlows regions flow left flow (fpAssignedNames st.raw)
  | _ =>
      let assigned := fpAssignedNames st.raw
      return (assigned.map (·, none) ++ flow.1.filter (fun p => !(assigned.contains p.1)),
        assigned.map (·, none) ++ flow.2.filter (fun p => !(assigned.contains p.1)))

end

private def fpProvenanceAfterStmt (regions : Inference.RegionDTypes)
    (pointers : FPPtrProvenance) (st : TSyntax `tritonStmt) : MacroM FPPtrProvenance := do
  return (← fpFlowAfterStmt regions (pointers, pointers) st).1

/-- Explicit wide storage provenance supplies load compute precision, independently
of the mathematical real carrier used by unannotated source kernels. -/
partial def annotateWideFloatingLoadStorage (regions : Inference.RegionDTypes)
    (pointers : FPPtrProvenance) (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args => do
        let args ← args.mapM (annotateWideFloatingLoadStorage regions pointers)
        pure (.node info kind args)
    | other => pure other
  let keyword? (p : TSyntax `tritonExpr) (kwargs : TSyntaxArray `tritonMemKwarg) :
      MacroM (Option (TSyntax `tritonMemKwarg)) := do
    for kw in kwargs do
      match kw with
      | `(tritonMemKwarg| $name:ident = $_:tritonDType)
      | `(tritonMemKwarg| $name:ident = $_:tritonExpr) =>
          if name.getId.eraseMacroScopes.toString == "dtype" then return none
      | _ => pure ()
    let dtype ← match ← fpPointerDType? regions pointers p with
      | some .real => `(tritonDType| tl.float64)
      | some .fp32 => `(tritonDType| tl.float32)
      | _ => return none
    let kwName : Ident := mkIdent `dtype
    return some (← `(tritonMemKwarg| $kwName:ident = $dtype:tritonDType))
  let expr : TSyntax `tritonExpr := ⟨raw⟩
  match expr with
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))).raw
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      let some kw ← keyword? p kwargs | return raw
      let kwargs := kwargs.push kw
      return (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))).raw
  | _ => return raw

/-- Annotate only unsigned-load casts at a statement's current program point.
Do not annotate whole branch bodies with pointer evidence from before them. -/
private partial def annotateMachineLoadCasts (regions : Inference.RegionDTypes)
    (pointers : Inference.PtrElems) (raw : Syntax) : MacroM Syntax := do
  let raw ← match raw with
    | .node info kind args =>
        pure (.node info kind (← args.mapM (annotateMachineLoadCasts regions pointers)))
    | other => pure other
  let expr : TSyntax `tritonExpr := ⟨raw⟩
  let dtype? := match expr with
    | `(tritonExpr| tl.load($_p:tritonExpr, $_mask:tritonExpr $[, $_kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => some dt
    | `(tritonExpr| tl.load($_p:tritonExpr $[, $_kwargs:tritonMemKwarg]*).to($dt:tritonDType)) => some dt
    | `(tritonExpr| tl.cast($_e:tritonExpr, $dt:tritonDType)) => some dt
    | _ => (methodCast? expr).map Prod.snd
  if let some dtype := dtype? then
    if (unsignedCastWidth? dtype).isSome || signedMachineCast dtype then
      return ← annotateFloatingLoadStorage regions pointers raw
  return raw

mutual

partial def expandStmt (env : Env) (pinned intPinned : List String)
    (regionDTypes : Inference.RegionDTypes) (ptrElems : Inference.PtrElems)
    (fpSource : FPSourceContext) (fpPtrElems : FPPtrProvenance)
    (facts : StaticBoolFacts) (stx : TSyntax `tritonStmt) :
    MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
  -- Normalize a value-dtype cast before assignment dispatch as well as before
  -- expression expansion. Otherwise direct-load parser alternatives can take
  -- the unannotated load path before seeing the explicit conversion.
  let stx ← match stx with
    | `(tritonStmt| $id:ident = $value:tritonExpr) => do
        let raw ← annotateValueDTypeCasts env fpSource.legacyPointerCasts value.raw
        if raw == value.raw then pure stx else
          let value : TSyntax `tritonExpr := ⟨raw⟩
          `(tritonStmt| $id:ident = $value:tritonExpr)
    | `(tritonStmt| $id:ident := $value:tritonExpr) => do
        let raw ← annotateValueDTypeCasts env fpSource.legacyPointerCasts value.raw
        if raw == value.raw then pure stx else
          let value : TSyntax `tritonExpr := ⟨raw⟩
          `(tritonStmt| $id:ident := $value:tritonExpr)
    | _ => pure stx
  let annotate (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
    let raw ← annotateValueDTypeCasts env fpSource.legacyPointerCasts e.raw
    let raw ← annotatePointerElementCasts fpSource.regions fpSource.legacyPointerCasts fpPtrElems raw
    return ⟨← annotateMachineLoadCasts regionDTypes ptrElems raw⟩
  let expandExpr (env : Env) (e : TSyntax `tritonExpr) : MacroM EOut := do
    expandExpr env (← annotate e)
  let expandBoolCondition (env : Env) (e : TSyntax `tritonExpr) : MacroM EOut := do
    expandBoolCondition env (← annotate e)
  let expandNatExpectedExpr (env : Env) (e : TSyntax `tritonExpr) : MacroM EOut := do
    withRef e <| expandNatExpectedExpr env (← annotate e)
  let expandIntExpectedExpr (env : Env) (e : TSyntax `tritonExpr) : MacroM EOut := do
    expandIntExpectedExpr env (← annotate e)
  let signedNames := intPinned ++ env.filterMap (fun entry =>
    if entry.2.1 == DInfo.int || entry.2.1 == DInfo.int8 ||
        entry.2.1 == DInfo.int32 then some entry.1 else none)
  let signedBounds (start stop : TSyntax `tritonExpr) : Bool :=
    Inference.isSignedExpr signedNames start || Inference.isSignedExpr signedNames stop
  let expandSignedRange (i : Ident) (start stop stride : TSyntax `tritonExpr)
      (stmts : TSyntaxArray `tritonStmt) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let nameLit ← identAsStr i
    let start' ← expandIntExpectedExpr env start
    let stop' ← expandIntExpectedExpr env stop
    let stride' ← expandNatExpectedExpr env stride
    ensureShape SInfo.scalar start'.shape "signed range start"
    ensureShape SInfo.scalar stop'.shape "signed range stop"
    ensureShape SInfo.scalar stride'.shape "signed range stride"
    let bodyEnv := (i.getId.toString, DInfo.int, SInfo.scalar, none) :: env
    let (algBody, computeBody, _, bodyHasCompute) ←
      expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
    pure (← `(Stmt.forRangeInt $nameLit $start'.term $stop'.term $stride'.term [$algBody,*]),
      ← `(ComputeStmt.forRangeInt $nameLit $start'.term $stop'.term $stride'.term [$computeBody,*]),
      env, bodyHasCompute)
  let unsupportedAtomic2 (op : String) (p v : TSyntax `tritonExpr) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    discard <| expandExpr env p
    discard <| expandExpr env v
    let opTerm : TSyntax `term := ⟨Syntax.mkStrLit op⟩
    pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
      ← `(ComputeStmt.effectMarker $opTerm), env, Bool.true)
  let ensureNoAtomicKwargs (op : String) (kwargs : TSyntaxArray `tritonMemKwarg) :
      MacroM Unit := do
    unless kwargs.isEmpty do
      Macro.throwError (op ++ ": kwargs are not modeled for return-valued atomics yet")
  let expandAtomicRMWCore (opName : String) (opTerm : TSyntax `term)
      (dest : Option Ident) (p input : TSyntax `tritonExpr)
      (extraInput : Option (TSyntax `tritonExpr)) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    -- An RMW's returned value comes from memory, not from its input operand.
    -- Operand storage provenance must not become provenance of that result.
    let input' := (← expandExpr env input).forgetStorageWidth
    ensureAlgorithmOnly opName input'
    let extra' ←
      match extraInput with
      | none => pure none
      | some extra => do
          let e := (← expandExpr env extra).forgetStorageWidth
          ensureAlgorithmOnly opName e
          ensureDType input'.dtype e.dtype opName
          pure (some e)
    let mkTerms (memTerm : TSyntax `term) (targetShape : SInfo) :
        MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
      let inputTerm ← coerceShape input'.term input'.shape targetShape opName
      let extraTerm? ←
        match extra' with
        | none => pure none
        | some e => pure (some (← coerceShape e.term e.shape targetShape opName))
      let dt ← input'.dtype.term
      let sh ← targetShape.term
      let destTerm ←
        match dest with
        | none => `(Option.none)
        | some ident =>
            let lit ← identAsStr ⟨ident.raw⟩
            `(Option.some $lit)
      let extraAlgTerm ←
        match extraTerm? with
        | none => `(Option.none)
        | some t => `(Option.some $t)
      let extraComputeTerm ←
        match extraTerm? with
        | none => `(Option.none)
        | some t => `(Option.some (ComputeExpr.alg $t))
      let alg ←
        `(Stmt.atomicRMW $opTerm $dt $sh $memTerm $inputTerm $extraAlgTerm
            (MaskOpt.none) $destTerm)
      let compute ←
        `(ComputeStmt.atomicRMW $opTerm $dt $sh $memTerm
            (ComputeExpr.alg $inputTerm) $extraComputeTerm (MaskOpt.none) $destTerm)
      let nextEnv :=
        match dest with
        | none => env
        | some ident => (ident.getId.toString, input'.dtype, targetShape, none) :: env
      pure (alg, compute, nextEnv, Bool.false)
    match ← expandStaticPtrExpr env p with
    | some sp =>
        mkTerms (← `(MemAccess.region $sp.region $sp.offset)) sp.shape
    | none =>
        let p' ← expandExpr env p
        ensureDType .ptr p'.dtype (opName ++ " pointer")
        mkTerms (← `(MemAccess.ptr $p'.term)) p'.shape
  let pointerDTypeHint (p : TSyntax `tritonExpr) : MacroM (Option DInfo) := do
    match ← expandStaticPtrExpr env p with
    | some sp =>
        match regionTermName sp.region with
        | some name => pure (Inference.lookupRegionDType regionDTypes name)
        | none => pure none
    | none =>
        pure (Inference.pointerElementDType? regionDTypes ptrElems p)
  let loadDTypeHint (dest : Ident) (p : TSyntax `tritonExpr) : MacroM (Option DInfo) := do
    let regionHint ← pointerDTypeHint p
    let pinHint : Option DInfo :=
      if intPinned.contains dest.getId.toString then some .int
      else if pinned.contains dest.getId.toString then some .nat
      else none
    pure (regionHint.orElse (fun _ => pinHint))
  -- The result of an int1 cast is Bool, but an explicitly integer mask buffer
  -- must still be read through the integer memory channel. As with tl.where,
  -- integer truthiness is a nonzero comparison, not a bool-typed load.
  let integerBoolLoadCast (p : TSyntax `tritonExpr)
      (kwargs : TSyntaxArray `tritonMemKwarg)
      (mask : Option (TSyntax `tritonExpr)) : MacroM (Option EOut) := do
    let hint ← pointerDTypeHint p
    if hint != some DInfo.int && hint != some DInfo.int32 then return none
    let loaded := (← expandLoad expandExpr expandStaticPtrExpr env p kwargs
      (defaultDType := hint) (positionalMask := mask)).forgetStorageWidth
    ensureDType .int loaded.dtype "integer load cast to int1"
    let zero ← coerceShape (← `(Op.constInt 0)) SInfo.scalar loaded.shape
      "integer load cast to int1"
    let (bc, _) ← broadcastTerm loaded.shape loaded.shape "integer load cast to int1"
    return some ⟨← `(Op.ne ComparableDType.int $bc $loaded.term $zero),
      .bool, loaded.shape, none, none⟩
  let integerBoolLoadCast? (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    match e with
    | `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int1))
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int1)) =>
        integerBoolLoadCast p kwargs (some mask)
    | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int1))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int1)) =>
        integerBoolLoadCast p kwargs none
    | _ => return none
  -- A natural-valued logical memory channel still needs its explicit signed
  -- widening. The destination of .to(int64) cannot retag the source load as
  -- an Int cell, even when later address uses pin its destination register.
  let naturalLoadWidening? (dest : Ident) (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    let load? ← match e with
      | `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int64))
      | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int64)) =>
          pure (some (p, kwargs, some mask))
      | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int64))
      | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int64)) =>
          pure (some (p, kwargs, none))
      | _ => pure none
    let some (p, kwargs, mask) := load? | return none
    let hint ← loadDTypeHint dest p
    if hint != some DInfo.nat then return none
    let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
      (defaultDType := hint) (positionalMask := mask)
    ensureDType .nat loaded.dtype "natural load widened to signed int64"
    return some ⟨← `(Op.castNatToInt $loaded.term), .int, loaded.shape, none, none⟩
  -- Preserve signed source storage and apply explicit i32 narrowing before
  -- assignment inference. Same-width i32 casts retain their existing AST.
  let signedLoadCast? (dest : Ident) (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    match e with
    | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int32))
    | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to(tl.int64))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int32))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to(tl.int64)) =>
        let hint ← loadDTypeHint dest p
        if hint == some DInfo.int || hint == some DInfo.int32 then
          let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
            (defaultDType := hint)
          let same32 := match e with
            | `(tritonExpr| (tl.load($_ $[, $_:tritonMemKwarg]*)).to(tl.int32))
            | `(tritonExpr| tl.load($_ $[, $_:tritonMemKwarg]*).to(tl.int32)) => Bool.true
            | _ => Bool.false
          if same32 then
            if let some cast ← machineIntegerCast? (← `(tritonDType| tl.int32)) loaded then
              return some cast
          return some loaded.forgetStorageWidth
        return none
    | _ => return none
  -- Read integer cells at their source dtype before applying a machine
  -- destination cast. In particular, int64 seeds remain signed loads.
  let machineLoadCast? (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    let cast? ← match e with
      | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
          pure (some (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)), dt))
      | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
          pure (some (← `(tritonExpr| tl.load($p $[, $kwargs:tritonMemKwarg]*)), dt))
      | `(tritonExpr| tl.cast($a:tritonExpr, $dt:tritonDType)) => pure (some (a, dt))
      | _ => pure (methodCast? e)
    let some (a, dt) := cast? | return none
    if (unsignedCastWidth? dt).isNone && !signedMachineCast dt then return none
    let rec loadInteger (a : TSyntax `tritonExpr) : MacroM (Option EOut) := do
      match a with
      | `(tritonExpr| ($inner:tritonExpr)) => loadInteger inner
      | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
          let hint ← pointerDTypeHint p
          if hint != some DInfo.int && hint != some DInfo.int32 &&
              hint != some DInfo.int8 && hint != some DInfo.nat then return none
          return ← machineIntegerCast? dt (← expandLoad expandExpr expandStaticPtrExpr
            env p kwargs (defaultDType := hint) (positionalMask := some mask))
      | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
          let hint ← pointerDTypeHint p
          if hint != some DInfo.int && hint != some DInfo.int32 &&
              hint != some DInfo.int8 && hint != some DInfo.nat then return none
          return ← machineIntegerCast? dt (← expandLoad expandExpr expandStaticPtrExpr
            env p kwargs (defaultDType := hint))
      | _ => return none
    loadInteger a
  -- Keep the pointer's known floating storage type through a load/cast
  -- assignment. The cast target describes the result, not the memory cell.
  -- Ordinary fp32/real loads already use the real carrier; preserve their
  -- existing expansion and handle the distinct storage channels here.
  let floatLoadCast (dest : Ident) (p : TSyntax `tritonExpr)
      (kwargs : TSyntaxArray `tritonMemKwarg) (mask : Option (TSyntax `tritonExpr))
      (dt : TSyntax `tritonDType) : MacroM (Option EOut) := do
    let hint ← loadDTypeHint dest p
    let dst ← expandDType dt
    match hint with
    | some .fp16 | some .bf16 | some .f8e4 | some .f8e5 | some (.floatVar _) => pure ()
    | some .real =>
        -- The mathematical real carrier alone is not fp64 storage evidence.
        -- Only explicit floating provenance enables this load/cast repair.
        if (← fpPointerDType? fpSource.regions fpPtrElems p) != some .real then return none
        if dst != .real && dst != .fp32 then return none
    | _ => return none
    match dst with
    | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ => pure ()
    | _ => return none
    let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
      (defaultDType := hint) (positionalMask := mask)
    let loaded ← if hint == some DInfo.real then do
      let ct ← fp64ComputeExpr loaded.term
      pure { loaded with computeTerm := some ct, computeDType? := some .real }
      else pure loaded
    return some (← explicitFloatCast dst loaded)
  let floatLoadCast? (dest : Ident) (e : TSyntax `tritonExpr) : MacroM (Option EOut) := do
    match e with
    | `(tritonExpr| (tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
        floatLoadCast dest p kwargs (some mask) dt
    | `(tritonExpr| (tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)).to($dt:tritonDType))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType)) =>
        floatLoadCast dest p kwargs none dt
    | _ => return none
  -- An in-place arithmetic update has the same load/cast operand as an
  -- ordinary assignment. Resolve its storage from the pointer at this program
  -- point; the target of `.to` must not become the load's storage dtype.
  let expandUpdateOperand (dest : Ident) (e : TSyntax `tritonExpr) : MacroM EOut := do
    if let some loaded ← machineLoadCast? e then return loaded
    if let some loaded ← floatLoadCast? dest e then return loaded
    if let some loaded ← naturalLoadWidening? dest e then return loaded
    if let some loaded ← signedLoadCast? dest e then return loaded
    -- A machine cast establishes the signed result before address-use
    -- inference. Keep its negative values until the final address projection.
    let signedCast := match e with
      | `(tritonExpr| tl.cast($_:tritonExpr, $dt:tritonDType)) => signedMachineCast dt
      | _ => (methodCast? e).any (fun pair => signedMachineCast pair.2)
    if signedCast then
      let inferred ← expandExpr env e
      if inferred.dtype == .int32 || inferred.dtype == .int then return inferred
    let e : TSyntax `tritonExpr := ⟨← annotateFloatingLoadStorage regionDTypes ptrElems e.raw⟩
    let e : TSyntax `tritonExpr := ⟨← annotateWideFloatingLoadStorage fpSource.regions fpPtrElems e.raw⟩
    expandExpr env e
  let expandAssignment (dest : Ident) (e : TSyntax `tritonExpr) : MacroM EOut := do
    -- This source operation must be materialized in a Bool register before
    -- algorithm-layer composition. Do not add an expandExpr fallback that
    -- would let where/if silently discard its compute-only child.
    match e with
    | `(tritonExpr| tl.__finite_helper_resolved($argument:tritonExpr)) =>
        let input ← expandExpr env argument
        if ← finiteHelperIsFP64 input then
          return ← explicitFiniteClassifier .fp64 input
        else
          -- Keep the helper's explicit cast even for an already-fp32 operand.
          return ← explicitFiniteClassifier .fp32 (← explicitFloatCast .fp32 input)
    | `(tritonExpr| tl.extra.cuda.libdevice.finitef($input:tritonExpr))
    | `(tritonExpr| libdevice.finitef($input:tritonExpr)) =>
        return ← explicitFiniteClassifier .fp32 (← expandExpr env input)
    | `(tritonExpr| tl.extra.cuda.libdevice.isfinited($input:tritonExpr))
    | `(tritonExpr| libdevice.isfinited($input:tritonExpr)) =>
        return ← explicitFiniteClassifier .fp64 (← expandExpr env input)
    | _ => pure ()
    if let some loaded ← machineLoadCast? e then return loaded
    if let some loaded ← integerBoolLoadCast? e then return loaded
    if let some loaded ← floatLoadCast? dest e then return loaded
    if let some loaded ← naturalLoadWidening? dest e then return loaded
    if let some loaded ← signedLoadCast? dest e then return loaded
    -- A machine cast establishes the signed result before address-use
    -- inference. Keep its negative values until the final address projection.
    let signedCast := match e with
      | `(tritonExpr| tl.cast($_:tritonExpr, $dt:tritonDType)) => signedMachineCast dt
      | _ => (methodCast? e).any (fun pair => signedMachineCast pair.2)
    if signedCast then
      let inferred ← expandExpr env e
      if inferred.dtype == .int32 || inferred.dtype == .int then return inferred
    let e : TSyntax `tritonExpr := ⟨← annotateFloatingLoadStorage regionDTypes ptrElems e.raw⟩
    let e : TSyntax `tritonExpr := ⟨← annotateWideFloatingLoadStorage fpSource.regions fpPtrElems e.raw⟩
    -- A name may first hold a packed integer and later its dequantized float.
    -- A whole-body address/bitwise pin must not override this explicit cast.
    let explicitConversion := containsExplicitConversion env e.raw
    if explicitConversion then
      let inferred ← expandExpr env e
      match inferred.dtype with
      -- A nested explicit signed cast/shift carries a known native width too.
      -- A later bitcast/address pin must not erase it before signed bitwise ops.
      | .int8 | .int32 | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .floatVar _ => return inferred
      | _ => pure ()
    if intPinned.contains dest.getId.toString then
      -- Signed offset propagation can also reach a named pointer binding.
      -- Keep its pointer dtype: only the offset is an integer expression.
      if (Inference.pointerElementDType? regionDTypes ptrElems e).isSome then
        let inferred ← expandExpr env e
        if inferred.dtype == DInfo.ptr then return inferred
      -- Source-width inference is stronger than the preliminary address pin.
      -- Exact signed casts, shifts and bitwise operations already carry the
      -- native width; retain it across ordinary assignments and copies.
      try
        let inferred ← expandExpr env e
        if inferred.dtype == .int8 || inferred.dtype == .int32 then return inferred
      catch _ => pure ()
      return ← expandIntExpectedExpr env e
    -- Propagate a signed register learned from a typed load, even when a
    -- later address use caused the preliminary scan to request Nat. Pointer
    -- expressions retain their ordinary pointer expansion.
    if Inference.isSignedExpr signedNames e then
      let inferred ← expandExpr env e
      if inferred.forgetStorageWidth.dtype == DInfo.int then return inferred
    if pinned.contains dest.getId.toString then return ← expandNatExpectedExpr env e
    expandExpr env e
  let preserveWideLoad (p : TSyntax `tritonExpr) (e : EOut) : MacroM EOut := do
    if e.dtype == .real && (← fpPointerDType? fpSource.regions fpPtrElems p) == some .real then
      let ct ← fp64ComputeExpr e.term
      return { e with computeTerm := some ct, computeDType? := some .real }
    return e
  let emitAssign (dest : Ident) (e' : EOut) : MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let nameLit ← identAsStr dest
    let dt ← e'.dtype.term
    let sh ← e'.shape.term
    let exprTerm ←
      match e'.computeTerm with
      | some ce => pure ce
      | none => `(ComputeExpr.alg $e'.term)
    pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
      ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
      (dest.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
      e'.computeTerm.isSome)
  let expandLoadSubAssign (dest : Ident) (p : TSyntax `tritonExpr)
      (kwargs : TSyntaxArray `tritonMemKwarg) (rhs : TSyntax `tritonExpr) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    let hint ← loadDTypeHint dest p
    let load' := (← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)).forgetStorageWidth
    ensureAlgorithmOnly "arithmetic" load'
    let rhs' ←
      match ← expandLeanAntiquoteAs? load'.dtype rhs with
      | some out => pure out
      | none => expandExpr env rhs
    let rhs' ← integralSourceLiteral load'.dtype rhs rhs'
    ensureAlgorithmOnly "arithmetic" rhs'
    -- Loaded storage widths do not change the mathematical integer carrier.
    -- Normalize both operands, including a previously bound int32 register,
    -- before comparing dtypes; mixed natural offsets widen to signed Int.
    let (load', rhs') ← coerceNatIntOperands "arithmetic" load' rhs'
    ensureDType load'.dtype rhs'.dtype "arithmetic"
    let np ← load'.dtype.numericProof
    let (bc, outShape) ← broadcastTerm load'.shape rhs'.shape "arithmetic"
    let term ← `(Op.sub $np $bc $load'.term $rhs'.term)
    emitAssign dest { term := term, dtype := load'.dtype, shape := outShape, computeTerm := none }
  let expandSelfAddAssign (dest old : Ident) (rhs : TSyntax `tritonExpr) :
      MacroM (TSyntax `term × TSyntax `term × Env × Bool) := do
    if dest.getId.getString! != old.getId.getString! then
      let sumExpr : TSyntax `tritonExpr ← `(tritonExpr| $old:ident + $rhs:tritonExpr)
      -- A fresh `dest = old + rhs` is an ordinary assignment. Use the same
      -- source-width and known-register inference as every other expression;
      -- preliminary Nat address pins cannot retag an already signed operand.
      let e' ← expandAssignment dest sumExpr
      emitAssign dest e'
    else
      let rawName := old.getId.toString
      let userName := old.getId.getString!
      let erasedName := old.getId.eraseMacroScopes.toString
      let some (envName, lhsDType, lhsShape, lhsComputeDType?) :=
          env.find? (fun entry =>
            entry.1 == rawName || entry.1 == userName || entry.1 == erasedName)
        | do
            let nameLit ← identAsStr dest
            let rhs' ← expandNatExpectedExpr env rhs
            ensureAlgorithmOnly "pointer rebind assignment" rhs'
            let region : TSyntax `term := ⟨old.raw⟩
            let lhsTerm ← `(Op.ptrBase $region)
            let (bc, outShape) ← broadcastTerm SInfo.scalar rhs'.shape "pointer rebind assignment"
            let outShapeTerm ← outShape.term
            let term ← `(Op.ptrAdd $bc $lhsTerm $rhs'.term)
            return (← `(Stmt.assign TileDType.ptr $outShapeTerm $nameLit $term),
              ← `(ComputeStmt.assign TileDType.ptr $outShapeTerm $nameLit (ComputeExpr.alg $term)),
              (dest.getId.toString, .ptr, outShape, none) :: env,
              Bool.false)
      let nameLit : TSyntax `term := Syntax.mkStrLit envName
      if lhsDType == .ptr then
        let rhs0 := (← expandExpr env rhs).forgetStorageWidth
        let rhs' ←
          match rhs0.dtype with
          | .nat => pure rhs0
          | .int =>
              pointerOffsetIntAsNat rhs0
          | _ => expandNatExpectedExpr env rhs
        ensureAlgorithmOnly "pointer rebind assignment" rhs'
        let lhsShapeTerm ← lhsShape.term
        let lhsTerm ← `(Op.ref TileDType.ptr $lhsShapeTerm $nameLit)
        let (bc, outShape) ← broadcastTerm lhsShape rhs'.shape "pointer rebind assignment"
        let outShapeTerm ← outShape.term
        let term ← `(Op.ptrAdd $bc $lhsTerm $rhs'.term)
        return (← `(Stmt.assign TileDType.ptr $outShapeTerm $nameLit $term),
          ← `(ComputeStmt.assign TileDType.ptr $outShapeTerm $nameLit (ComputeExpr.alg $term)),
          (dest.getId.toString, .ptr, outShape, none) ::
            (envName, .ptr, outShape, none) :: env,
          Bool.false)
      let lhsShapeTerm ← lhsShape.term
      let lhsDTypeTerm ← lhsDType.term
      let lhsTerm ← `(Op.ref $lhsDTypeTerm $lhsShapeTerm $nameLit)
      let lhsComputeTerm? ←
        match lhsComputeDType? with
        | some .fp32 => pure (some (← fp32ComputeExpr lhsTerm))
        | some .real => pure (some (← fp64ComputeExpr lhsTerm))
        | some _ =>
            Macro.throwError
              ("identifier `" ++ envName ++ "` has unsupported compute dtype annotation")
        | none => pure none
      let lhs : EOut :=
        { term := lhsTerm, dtype := lhsDType, shape := lhsShape,
          computeTerm := lhsComputeTerm?, computeDType? := lhsComputeDType? }
      let rhs0 ←
        if lhsDType == .nat then
          expandNatExpectedExpr env rhs
        else
          expandExpr env rhs
      let rhs' ←
        if lhs.dtype == .real && rhs0.dtype == .nat then
          match ← expandLeanAntiquoteAs? .real rhs with
          | some out => pure out
          | none => pure rhs0
        else
          pure rhs0
      let (lhs, rhs') ← coerceNatIntOperands "self add assignment" lhs rhs'
      let (lhs, rhs') ← coerceRealArithOperands "self add assignment" lhs rhs'
      ensureComputeArithComposable "self add assignment lhs" lhs
      ensureComputeArithComposable "self add assignment rhs" rhs'
      unless lhs.dtype == rhs'.dtype do
        Macro.throwError "self add assignment: dtype mismatch"
      let np ← lhs.dtype.numericProof
      let (bc, outShape) ← broadcastTerm lhs.shape rhs'.shape "self add assignment"
      let eTerm ← `(Op.add $np $bc $lhs.term $rhs'.term)
      let (computeTerm?, computeDType?) ← floatingComputeArith? "self add assignment" eTerm lhs rhs'
      let e' : EOut :=
        { term := eTerm, dtype := lhs.dtype, shape := outShape,
          computeTerm := computeTerm?, computeDType? := computeDType? }
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (dest.getId.toString, e'.dtype, e'.shape, e'.computeDType?) ::
          (envName, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  match stx with
  | `(tritonStmt| $r0:ident, $r1:ident, $r2:ident, $r3:ident =
      tl.__philox_profiled($cfg:term, $seed:tritonExpr, $c0:tritonExpr,
        $c1:tritonExpr, $c2:tritonExpr, $c3:tritonExpr)) => do
      let some types := RandomProfile.directConfigTypes? cfg |
        Macro.throwError "tl.philox: malformed internal source profile"
      let operands := #[seed, c0, c1, c2, c3]
      let mut expanded : Array EOut := #[]
      for index in [:5] do
        expanded := expanded.push (← expandRandIntegerExpr env types[index]! operands[index]!)
      expandDirectPhilox env #[r0, r1, r2, r3] cfg 10 expanded
  | `(tritonStmt| $_:ident, $_:ident, $_:ident, $_:ident =
      tl.philox($_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:tritonExpr, $_:tritonExpr)) =>
      Macro.throwError "tl.philox requires source-profile preprocessing in a triton block"
  | `(tritonStmt| $valueName:ident, $indexName:ident = tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let original ← expandExpr env e
      -- Triton 3.5.1 promotes bf16 before either max branch; fp16 is
      -- retained specifically when indices are requested.
      let e' ← if original.dtype == .bf16 then do
          let term ← realMathTerm "tl.max return_indices" original
          pure (⟨term, .real, original.shape, some (← fp32ComputeExpr term), some .fp32⟩ : EOut)
        else pure original
      match e'.dtype with
      | .real | .fp32 | .fp16 | .f8e4 | .f8e5 => pure ()
      | _ => Macro.throwError "tl.max return_indices: unsupported or unspecialized storage dtype"
      let outDType ← e'.dtype.term
      let dims := match e'.shape with | .dims ds => ds
      let axisIdx ← parseMaxReturnIndicesKwargs dims kwargs
      let outDims ← eraseNth dims axisIdx
      let outShape := SInfo.dims outDims
      let outShapeTerm ← outShape.term
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      let valueLit ← identAsStr valueName
      let indexLit ← identAsStr indexName
      let (valueOp, indexOp) ← if e'.dtype == .real then
          let value ← if e'.computeDType?.isSome then
              `(Op.indexedMaxValue FloatDType.real (⟨$axisLit, by simp⟩) $e'.term)
            else `(Op.reduceMax (⟨$axisLit, by simp⟩) Bool.false $e'.term)
          let index ← match e'.computeDType? with
            | some .fp32 => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp32))
            | some .real => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp64))
            | _ => `(Op.argMax (⟨$axisLit, by simp⟩) $e'.term)
          pure (value, index)
        else do
          let fp ← e'.dtype.floatProof
          pure (← `(Op.indexedMaxValue $fp (⟨$axisLit, by simp⟩) $e'.term),
            ← `(Op.argMaxTyped $fp (⟨$axisLit, by simp⟩) $e'.term))
      let valueStmt ← `(Stmt.assign $outDType $outShapeTerm $valueLit $valueOp)
      let indexStmt ← `(Stmt.assign TileDType.nat $outShapeTerm $indexLit $indexOp)
      let (valueExpr?, precision?) ← floatingComputeArith? "tl.max return_indices" valueOp e' e'
      let valueExpr ← match valueExpr? with
        | some x => pure x
        | none => `(ComputeExpr.alg $valueOp)
      let indexExpr ← `(ComputeExpr.alg $indexOp)
      let valueCompute ← `(ComputeStmt.assign $outDType $outShapeTerm $valueLit $valueExpr)
      let indexCompute ← `(ComputeStmt.assign TileDType.nat $outShapeTerm $indexLit $indexExpr)
      let env' :=
        (indexName.getId.eraseMacroScopes.toString, DInfo.nat, outShape, none) ::
        (indexName.getId.getString!, DInfo.nat, outShape, none) ::
          (indexName.getId.toString, DInfo.nat, outShape, none) ::
          (valueName.getId.eraseMacroScopes.toString, e'.dtype, outShape, precision?) ::
          (valueName.getId.getString!, e'.dtype, outShape, precision?) ::
          (valueName.getId.toString, e'.dtype, outShape, precision?) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$valueStmt, $indexStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$valueCompute, $indexCompute]),
        env', valueExpr?.isSome)
  | `(tritonStmt| $valueName:ident, $indexName:ident := tl.max($e:tritonExpr $[, $kwargs:tritonReduceKwarg]*)) => do
      let original ← expandExpr env e
      -- Triton 3.5.1 promotes bf16 before either max branch; fp16 is
      -- retained specifically when indices are requested.
      let e' ← if original.dtype == .bf16 then do
          let term ← realMathTerm "tl.max return_indices" original
          pure (⟨term, .real, original.shape, some (← fp32ComputeExpr term), some .fp32⟩ : EOut)
        else pure original
      match e'.dtype with
      | .real | .fp32 | .fp16 | .f8e4 | .f8e5 => pure ()
      | _ => Macro.throwError "tl.max return_indices: unsupported or unspecialized storage dtype"
      let outDType ← e'.dtype.term
      let dims := match e'.shape with | .dims ds => ds
      let axisIdx ← parseMaxReturnIndicesKwargs dims kwargs
      let outDims ← eraseNth dims axisIdx
      let outShape := SInfo.dims outDims
      let outShapeTerm ← outShape.term
      let axisLit : TSyntax `num := ⟨Syntax.mkNumLit (toString axisIdx)⟩
      let valueLit ← identAsStr valueName
      let indexLit ← identAsStr indexName
      let (valueOp, indexOp) ← if e'.dtype == .real then
          let value ← if e'.computeDType?.isSome then
              `(Op.indexedMaxValue FloatDType.real (⟨$axisLit, by simp⟩) $e'.term)
            else `(Op.reduceMax (⟨$axisLit, by simp⟩) Bool.false $e'.term)
          let index ← match e'.computeDType? with
            | some .fp32 => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp32))
            | some .real => `(Op.argMaxTyped FloatDType.real (⟨$axisLit, by simp⟩) $e'.term (some IndexComputePrecision.fp64))
            | _ => `(Op.argMax (⟨$axisLit, by simp⟩) $e'.term)
          pure (value, index)
        else do
          let fp ← e'.dtype.floatProof
          pure (← `(Op.indexedMaxValue $fp (⟨$axisLit, by simp⟩) $e'.term),
            ← `(Op.argMaxTyped $fp (⟨$axisLit, by simp⟩) $e'.term))
      let valueStmt ← `(Stmt.assign $outDType $outShapeTerm $valueLit $valueOp)
      let indexStmt ← `(Stmt.assign TileDType.nat $outShapeTerm $indexLit $indexOp)
      let (valueExpr?, precision?) ← floatingComputeArith? "tl.max return_indices" valueOp e' e'
      let valueExpr ← match valueExpr? with
        | some x => pure x
        | none => `(ComputeExpr.alg $valueOp)
      let indexExpr ← `(ComputeExpr.alg $indexOp)
      let valueCompute ← `(ComputeStmt.assign $outDType $outShapeTerm $valueLit $valueExpr)
      let indexCompute ← `(ComputeStmt.assign TileDType.nat $outShapeTerm $indexLit $indexExpr)
      let env' :=
        (indexName.getId.eraseMacroScopes.toString, DInfo.nat, outShape, none) ::
        (indexName.getId.getString!, DInfo.nat, outShape, none) ::
          (indexName.getId.toString, DInfo.nat, outShape, none) ::
          (valueName.getId.eraseMacroScopes.toString, e'.dtype, outShape, precision?) ::
          (valueName.getId.getString!, e'.dtype, outShape, precision?) ::
          (valueName.getId.toString, e'.dtype, outShape, precision?) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$valueStmt, $indexStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$valueCompute, $indexCompute]),
        env', valueExpr?.isSome)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident = tl.broadcast($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      let aOut ←
        if a'.shape.eq b'.shape then
          pure a'
        else
          let aDims := match a'.shape with | .dims ds => ds
          let bDims := match b'.shape with | .dims ds => ds
          if aDims.length + 1 == bDims.length && termListEq aDims (bDims.drop 1) then
            expandExpandDims expandExpr env a (axisIdx := 0)
          else
            Macro.throwError "tl.broadcast tuple: only same-rank or one leading broadcast axis is supported"
      let mkAssign (name : Ident) (e' : EOut) : MacroM (TSyntax `term × TSyntax `term) := do
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
          ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
      let (aStmt, aCompute) ← mkAssign lhs0 aOut
      let (bStmt, bCompute) ← mkAssign lhs1 b'
      let env' :=
        (lhs1.getId.toString, b'.dtype, b'.shape, b'.computeDType?) ::
        (lhs0.getId.toString, aOut.dtype, aOut.shape, aOut.computeDType?) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$aStmt, $bStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$aCompute, $bCompute]),
        env', aOut.computeTerm.isSome || b'.computeTerm.isSome)
  | `(tritonStmt| $lhs0:ident, _ = tl.broadcast($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      let aOut ←
        if a'.shape.eq b'.shape then
          pure a'
        else
          let aDims := match a'.shape with | .dims ds => ds
          let bDims := match b'.shape with | .dims ds => ds
          if aDims.length + 1 == bDims.length && termListEq aDims (bDims.drop 1) then
            expandExpandDims expandExpr env a (axisIdx := 0)
          else
            Macro.throwError "tl.broadcast tuple: only same-rank or one leading broadcast axis is supported"
      let nameLit ← identAsStr lhs0
      let dt ← aOut.dtype.term
      let sh ← aOut.shape.term
      let exprTerm ←
        match aOut.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $aOut.term)
      let stmt ← `(Stmt.assign $dt $sh $nameLit $aOut.term)
      let compute ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm)
      let env' := (lhs0.getId.toString, aOut.dtype, aOut.shape, aOut.computeDType?) :: env
      pure (stmt, compute, env', aOut.computeTerm.isSome)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident := tl.broadcast($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      let aOut ←
        if a'.shape.eq b'.shape then
          pure a'
        else
          let aDims := match a'.shape with | .dims ds => ds
          let bDims := match b'.shape with | .dims ds => ds
          if aDims.length + 1 == bDims.length && termListEq aDims (bDims.drop 1) then
            expandExpandDims expandExpr env a (axisIdx := 0)
          else
            Macro.throwError "tl.broadcast tuple: only same-rank or one leading broadcast axis is supported"
      let mkAssign (name : Ident) (e' : EOut) : MacroM (TSyntax `term × TSyntax `term) := do
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
          ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
      let (aStmt, aCompute) ← mkAssign lhs0 aOut
      let (bStmt, bCompute) ← mkAssign lhs1 b'
      let env' :=
        (lhs1.getId.toString, b'.dtype, b'.shape, b'.computeDType?) ::
        (lhs0.getId.toString, aOut.dtype, aOut.shape, aOut.computeDType?) :: env
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$aStmt, $bStmt]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$aCompute, $bCompute]),
        env', aOut.computeTerm.isSome || b'.computeTerm.isSome)
  | `(tritonStmt| $lhs0:ident, _ := tl.broadcast($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr env a
      let b' ← expandExpr env b
      let aOut ←
        if a'.shape.eq b'.shape then
          pure a'
        else
          let aDims := match a'.shape with | .dims ds => ds
          let bDims := match b'.shape with | .dims ds => ds
          if aDims.length + 1 == bDims.length && termListEq aDims (bDims.drop 1) then
            expandExpandDims expandExpr env a (axisIdx := 0)
          else
            Macro.throwError "tl.broadcast tuple: only same-rank or one leading broadcast axis is supported"
      let nameLit ← identAsStr lhs0
      let dt ← aOut.dtype.term
      let sh ← aOut.shape.term
      let exprTerm ←
        match aOut.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $aOut.term)
      let stmt ← `(Stmt.assign $dt $sh $nameLit $aOut.term)
      let compute ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm)
      let env' := (lhs0.getId.toString, aOut.dtype, aOut.shape, aOut.computeDType?) :: env
      pure (stmt, compute, env', aOut.computeTerm.isSome)
  | `(tritonStmt| $outMean:ident, $outM2:ident, $outWeight:ident = triton_helpers.welford_reduce($x:tritonExpr, $mean:tritonExpr, $m2:tritonExpr, $weight:tritonExpr, $first:tritonExpr)) => do
      let values ← #[x, mean, m2, weight].mapM (expandExpr env)
      for value in values do
        ensureDType .real value.dtype "welford_reduce"
        ensureShape values[0]!.shape value.shape "welford_reduce"
        unless value.computeDType? == some .fp32 do
          Macro.throwError "welford_reduce requires explicit FP32 operands"
      let targets := [outMean.getId.toString, outM2.getId.toString, outWeight.getId.toString]
      unless targets.eraseDups.length == 3 do
        Macro.throwError "welford_reduce requires three distinct targets"
      let rec mentionsReduce (stx : Syntax) : Bool :=
        (stx.isIdent && targets.contains stx.getId.toString) || stx.getArgs.any mentionsReduce
      if (#[x, mean, m2, weight, first]).any (fun expr => mentionsReduce expr.raw) then
        Macro.throwError "welford_reduce targets must be separate from its input expressions"
      let .dims dims := values[0]!.shape
      let shape ← dims.toArray.mapM (fun dim => `(tritonExpr| $($dim:term)))
      let dtypeName := mkIdent `dtype
      let (alg, compute, nextEnv, hasCompute) ←
        expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $first:tritonExpr {
          $outWeight:ident = tl.full([$shape,*], 1.0, $dtypeName:ident=tl.float32)
          $outMean:ident = $x:tritonExpr
          $outM2:ident = tl.zeros([$shape,*], tl.float32)
        } else {
          $outWeight:ident = $weight:tritonExpr + 1.0
          $outMean:ident = $mean:tritonExpr + ($x:tritonExpr - $mean:tritonExpr) / $outWeight:ident
          $outM2:ident = $m2:tritonExpr + ($x:tritonExpr - $mean:tritonExpr) * ($x:tritonExpr - $outMean:ident)
        }))
      -- Shape annotations emitted through the inline source can contain an
      -- extra `(dim : Nat)` wrapper. Both branches have the checked input
      -- shape definitionally; keep that original macro-time shape key.
      let nextEnv := targets.foldl (fun result name =>
        (name, DInfo.real, values[0]!.shape, some DInfo.fp32) :: result) nextEnv
      pure (alg, compute, nextEnv, hasCompute)
  | `(tritonStmt| $outMean:ident, $outM2:ident, $outWeight:ident = triton_helpers.welford($mean:tritonExpr, $m2:tritonExpr, $weight:tritonExpr, 1, tree=$($tree:term))) => do
      let values ← #[mean, m2, weight].mapM (expandExpr env)
      for value in values do
        ensureDType .real value.dtype "welford"
        ensureShape values[0]!.shape value.shape "welford"
        unless value.computeDType? == some .fp32 do
          Macro.throwError "welford requires explicit FP32 operands"
      let .dims [row, _] := values[0]!.shape
        | Macro.throwError "welford requires [1, N] tiles and axis 1"
      unless termIsOne row do
        Macro.throwError "welford requires [1, N] tiles and axis 1"
      let targets := [outMean.getId.toString, outM2.getId.toString, outWeight.getId.toString]
      unless targets.eraseDups.length == 3 do
        Macro.throwError "welford requires three distinct targets"
      let rec mentionsTree (stx : Syntax) : Bool :=
        (stx.isIdent && targets.contains stx.getId.toString) || stx.getArgs.any mentionsTree
      if (#[mean, m2, weight]).any (fun expr => mentionsTree expr.raw) then
        Macro.throwError "welford targets must be separate from its input expressions"
      let meanTerm ← values[0]!.scopedFloatingTerm
      let m2Term ← values[1]!.scopedFloatingTerm
      let weightTerm ← values[2]!.scopedFloatingTerm
      let triple ← `(WelfordSource.expressions $meanTerm $m2Term $weightTerm $tree)
      let exprs : Array (TSyntax `term) := #[← `(Op.full [1] ($triple).mean),
        ← `(Op.full [1] ($triple).m2), ← `(Op.full [1] ($triple).weight)]
      let mut alg : Array (TSyntax `term) := #[]
      let mut compute : Array (TSyntax `term) := #[]
      let mut nextEnv := env
      for (name, expr) in #[outMean, outM2, outWeight].zip exprs do
        let nameLit ← identAsStr name
        alg := alg.push (← `(Stmt.assign .real [1] $nameLit $expr))
        compute := compute.push (← `(ComputeStmt.assign TileDType.real [1] $nameLit
          (ComputeExpr.compute (ComputeOp.alg ComputeDType.fp32 $expr))))
        nextEnv := (name.getId.toString, DInfo.real, .dims [← `(1)], some DInfo.fp32) :: nextEnv
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$alg,*]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$compute,*]), nextEnv, Bool.true)
  | `(tritonStmt| $_outMean:ident, $_outM2:ident, $_outWeight:ident = triton_helpers.welford($_mean:tritonExpr, $_m2:tritonExpr, $_weight:tritonExpr, $_axis:num, tree=$($_tree:term))) =>
      Macro.throwError "welford requires [1, N] tiles and axis 1"
  | `(tritonStmt| $_lhs0:ident, $_lhs1:ident = $_fn:ident($_args:tritonExpr,*)) => do
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) []),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) []),
        env, Bool.false)
  | `(tritonStmt| $_lhs0:ident, $_lhs1:ident, $_lhs2:ident = $_fn:ident($_args:tritonExpr,*)) => do
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) []),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) []),
        env, Bool.false)
  | `(tritonStmt| $_lhs0:ident, $_lhs1:ident, _ = $_fn:ident($_args:tritonExpr,*)) => do
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) []),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) []),
        env, Bool.false)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* = $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) => do
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      unless lhs.size == rhs.size do
        Macro.throwError
          ("multiple assignment: left side has " ++ toString lhs.size ++
           " targets but right side has " ++ toString rhs.size ++ " values")
      let mut algStmts : Array (TSyntax `term) := #[]
      let mut computeStmts : Array (TSyntax `term) := #[]
      let mut env' := env
      let mut hasCompute := Bool.false
      let mut expanded : Array EOut := #[]
      for expr in rhs do
        let e' ← expandExpr env expr
        expanded := expanded.push e'
        hasCompute := hasCompute || e'.computeTerm.isSome
      for idx in [:lhs.size] do
        let name := lhs[idx]!
        let e' := expanded[idx]!
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        algStmts := algStmts.push (← `(Stmt.assign $dt $sh $nameLit $e'.term))
        computeStmts := computeStmts.push
          (← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
        env' := (name.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env'
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$algStmts,*]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$computeStmts,*]),
        env', hasCompute)
  | `(tritonStmt| $lhs0:ident, $lhs1:ident $[, $lhsRest:ident]* := $rhs0:tritonExpr, $rhs1:tritonExpr $[, $rhsRest:tritonExpr]*) => do
      let lhs := #[lhs0, lhs1] ++ lhsRest
      let rhs := #[rhs0, rhs1] ++ rhsRest
      unless lhs.size == rhs.size do
        Macro.throwError
          ("multiple assignment: left side has " ++ toString lhs.size ++
           " targets but right side has " ++ toString rhs.size ++ " values")
      let mut algStmts : Array (TSyntax `term) := #[]
      let mut computeStmts : Array (TSyntax `term) := #[]
      let mut env' := env
      let mut hasCompute := Bool.false
      let mut expanded : Array EOut := #[]
      for expr in rhs do
        let e' ← expandExpr env expr
        expanded := expanded.push e'
        hasCompute := hasCompute || e'.computeTerm.isSome
      for idx in [:lhs.size] do
        let name := lhs[idx]!
        let e' := expanded[idx]!
        let nameLit ← identAsStr name
        let dt ← e'.dtype.term
        let sh ← e'.shape.term
        let exprTerm ←
          match e'.computeTerm with
          | some ce => pure ce
          | none => `(ComputeExpr.alg $e'.term)
        algStmts := algStmts.push (← `(Stmt.assign $dt $sh $nameLit $e'.term))
        computeStmts := computeStmts.push
          (← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm))
        env' := (name.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env'
      pure (← `(Stmt.ifThen (Op.constBool Bool.true) [$algStmts,*]),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.true) [$computeStmts,*]),
        env', hasCompute)
  | `(tritonStmt| $i:ident = tl.max($e:tritonExpr, $n:num)) => do
      let nameLit ← identAsStr i
      let e' ← expandReduceMax expandExpr env e
        #[← `(tritonReduceKwarg| $n:num)]
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      let env' :=
        (i.getId.eraseMacroScopes.toString, e'.dtype, e'.shape, e'.computeDType?) ::
        (i.getId.getString!, e'.dtype, e'.shape, e'.computeDType?) ::
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        env', e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident := tl.max($e:tritonExpr, $n:num)) => do
      let nameLit ← identAsStr i
      let e' ← expandReduceMax expandExpr env e
        #[← `(tritonReduceKwarg| $n:num)]
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      let env' :=
        (i.getId.eraseMacroScopes.toString, e'.dtype, e'.shape, e'.computeDType?) ::
        (i.getId.getString!, e'.dtype, e'.shape, e'.computeDType?) ::
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        env', e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident := tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) (some i) p v none
  | `(tritonStmt| $i:ident := tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) (some i) p cmp (some v)
  | `(tritonStmt| $i:ident = tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) (some i) p v none
  | `(tritonStmt| $i:ident = tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) (some i) p cmp (some v)
  | `(tritonStmt| $i:ident := tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*) - $rhs:tritonExpr) => do
      expandLoadSubAssign i p kwargs rhs
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*) - $rhs:tritonExpr) => do
      expandLoadSubAssign i p kwargs rhs
  | `(tritonStmt| $i:ident := tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let hint ← loadDTypeHint i p
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := hint) (positionalMask := some mask)
      emitAssign i (← preserveWideLoad p loaded)
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let hint ← loadDTypeHint i p
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs
        (defaultDType := hint) (positionalMask := some mask)
      emitAssign i (← preserveWideLoad p loaded)
  | `(tritonStmt| $i:ident := tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let nameLit ← identAsStr i
      let hint ← loadDTypeHint i p
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
      let e' ← preserveWideLoad p loaded
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident = tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      let nameLit ← identAsStr i
      let hint ← loadDTypeHint i p
      let loaded ← expandLoad expandExpr expandStaticPtrExpr env p kwargs (defaultDType := hint)
      let e' ← preserveWideLoad p loaded
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident = $j:ident + $rhs:tritonExpr) => do
      expandSelfAddAssign i j rhs
  | `(tritonStmt| $i:ident := $j:ident + $rhs:tritonExpr) => do
      expandSelfAddAssign i j rhs
  | `(tritonStmt| $i:ident % $e:tritonExpr) => do
      discard <| expandExpr env (← `(tritonExpr| $i:ident % $e:tritonExpr))
      let alg ← `(Stmt.ifThen (Op.constBool Bool.true) ([] : List Stmt))
      let compTerm ←
        `(ComputeStmt.ifThen (Op.constBool Bool.true) ([] : List ComputeStmt))
      pure (alg, compTerm, env, Bool.false)
  | `(tritonStmt| $i:ident := $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandAssignment i e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident = $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let e' ← expandAssignment i e
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (i.getId.toString, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident += $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let name := i.getId.toString
      if env.any (fun entry => entry.1 == name) then
        let (lhsDType, lhsShape) ← lookupEnv env name
        match lhsDType with
        | .ptr =>
            let rhs0 := (← expandExpr env e).forgetStorageWidth
            let rhs ←
              match rhs0.dtype with
              | .nat => pure rhs0
              | .int =>
                  pointerOffsetIntAsNat rhs0
              | _ => expandNatExpectedExpr env e
            ensureAlgorithmOnly "`+=` pointer offset" rhs
            let lhsShapeTerm ← lhsShape.term
            let lhsTerm ← `(Op.ref TileDType.ptr $lhsShapeTerm $nameLit)
            let (bc, outShape) ← broadcastTerm lhsShape rhs.shape "`+=` pointer offset"
            let outShapeTerm ← outShape.term
            let term ← `(Op.ptrAdd $bc $lhsTerm $rhs.term)
            pure (← `(Stmt.assign TileDType.ptr $outShapeTerm $nameLit $term),
              ← `(ComputeStmt.assign TileDType.ptr $outShapeTerm $nameLit (ComputeExpr.alg $term)),
              (name, .ptr, outShape, none) :: env,
              Bool.false)
        | _ =>
            let lhsShapeTerm ← lhsShape.term
            let lhsDTypeTerm ← lhsDType.term
            let lhsTerm ← `(Op.ref $lhsDTypeTerm $lhsShapeTerm $nameLit)
            let lhsComputeDType? := lookupComputeDType? env name
            let lhsComputeTerm? ←
              match lhsComputeDType? with
              | some .fp32 => pure (some (← fp32ComputeExpr lhsTerm))
              | some .real => pure (some (← fp64ComputeExpr lhsTerm))
              | some _ =>
                  Macro.throwError
                    ("identifier `" ++ name ++ "` has unsupported compute dtype annotation")
              | none => pure none
            let lhs : EOut :=
              { term := lhsTerm, dtype := lhsDType, shape := lhsShape,
                computeTerm := lhsComputeTerm?, computeDType? := lhsComputeDType? }
            let rhs0 ← integralSourceLiteral lhs.dtype e (← expandUpdateOperand i e)
            let rhs ←
              if lhs.dtype == .real && rhs0.dtype == .nat then
                match ← expandLeanAntiquoteAs? .real e with
                | some out => pure out
                | none => pure rhs0
              else
                pure rhs0
            let (lhs, rhs) ← coerceNatIntOperands "`+=`" lhs rhs
            let (lhs, rhs) ← coerceRealArithOperands "`+=`" lhs rhs
            ensureComputeArithComposable "`+=` lhs" lhs
            ensureComputeArithComposable "`+=` rhs" rhs
            unless lhs.dtype == rhs.dtype do
              Macro.throwError "`+=`: dtype mismatch"
            let np ← lhs.dtype.numericProof
            let (bc, outShape) ← broadcastTerm lhs.shape rhs.shape "`+=`"
            let eTerm ← `(Op.add $np $bc $lhs.term $rhs.term)
            let (computeTerm?, computeDType?) ← floatingComputeArith? "`+=`" eTerm lhs rhs
            let e' : EOut :=
              { term := eTerm, dtype := lhs.dtype, shape := outShape,
                computeTerm := computeTerm?, computeDType? := computeDType? }
            let dt ← e'.dtype.term
            let sh ← e'.shape.term
            let exprTerm ←
              match e'.computeTerm with
              | some ce => pure ce
              | none => `(ComputeExpr.alg $e'.term)
            pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
              ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
              (name, e'.dtype, e'.shape, e'.computeDType?) :: env,
              e'.computeTerm.isSome)
      else
          let rhs0 := (← expandExpr env e).forgetStorageWidth
          let rhs ←
            match rhs0.dtype with
            | .nat => pure rhs0
            | .int =>
                pointerOffsetIntAsNat rhs0
            | _ => expandNatExpectedExpr env e
          ensureAlgorithmOnly "`+=` pointer offset" rhs
          let region : TSyntax `term := ⟨i.raw⟩
          let lhsTerm ← `(Op.ptrBase $region)
          let (bc, outShape) ← broadcastTerm SInfo.scalar rhs.shape "`+=` pointer offset"
          let outShapeTerm ← outShape.term
          let term ← `(Op.ptrAdd $bc $lhsTerm $rhs.term)
          pure (← `(Stmt.assign TileDType.ptr $outShapeTerm $nameLit $term),
            ← `(ComputeStmt.assign TileDType.ptr $outShapeTerm $nameLit (ComputeExpr.alg $term)),
            (name, .ptr, outShape, none) :: env,
            Bool.false)
  | `(tritonStmt| $i:ident -= $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let name := i.getId.toString
      unless env.any (fun entry => entry.1 == name) do
        Macro.throwError ("`-=`: unknown identifier `" ++ name ++ "`")
      let (lhsDType, lhsShape) ← lookupEnv env name
      if lhsDType == .ptr then
        let rhs0 := (← expandExpr env e).forgetStorageWidth
        let rhs ←
          match rhs0.dtype with
          | .nat => pure rhs0
          | .int =>
              pointerOffsetIntAsNat rhs0
          | _ => expandNatExpectedExpr env e
        ensureAlgorithmOnly "`-=` pointer offset" rhs
        let lhsShapeTerm ← lhsShape.term
        let lhsTerm ← `(Op.ref TileDType.ptr $lhsShapeTerm $nameLit)
        let (bc, outShape) ← broadcastTerm lhsShape rhs.shape "`-=` pointer offset"
        let outShapeTerm ← outShape.term
        let term ← `(Op.ptrSub $bc $lhsTerm $rhs.term)
        pure (← `(Stmt.assign TileDType.ptr $outShapeTerm $nameLit $term),
          ← `(ComputeStmt.assign TileDType.ptr $outShapeTerm $nameLit (ComputeExpr.alg $term)),
          (name, .ptr, outShape, none) :: env,
          Bool.false)
      else if lhsDType == .blockPtr then
        Macro.throwError "`-=`: block pointer subtraction is not supported"
      else
      let lhsShapeTerm ← lhsShape.term
      let lhsDTypeTerm ← lhsDType.term
      let lhsTerm ← `(Op.ref $lhsDTypeTerm $lhsShapeTerm $nameLit)
      let lhsComputeDType? := lookupComputeDType? env name
      let lhsComputeTerm? ←
        match lhsComputeDType? with
        | some .fp32 => pure (some (← fp32ComputeExpr lhsTerm))
        | some .real => pure (some (← fp64ComputeExpr lhsTerm))
        | some _ =>
            Macro.throwError
              ("identifier `" ++ name ++ "` has unsupported compute dtype annotation")
        | none => pure none
      let lhs : EOut :=
        { term := lhsTerm, dtype := lhsDType, shape := lhsShape,
          computeTerm := lhsComputeTerm?, computeDType? := lhsComputeDType? }
      let rhs0 ← integralSourceLiteral lhs.dtype e (← expandUpdateOperand i e)
      let rhs ←
        if lhs.dtype == .real && rhs0.dtype == .nat then
          match ← expandLeanAntiquoteAs? .real e with
          | some out => pure out
          | none => pure rhs0
        else
          pure rhs0
      let (lhs, rhs) ← coerceNatIntOperands "`-=`" lhs rhs
      let (lhs, rhs) ← coerceRealArithOperands "`-=`" lhs rhs
      ensureComputeArithComposable "`-=` lhs" lhs
      ensureComputeArithComposable "`-=` rhs" rhs
      unless lhs.dtype == rhs.dtype do
        Macro.throwError "`-=`: dtype mismatch"
      let np ← lhs.dtype.numericProof
      let (bc, outShape) ← broadcastTerm lhs.shape rhs.shape "`-=`"
      let eTerm ← `(Op.sub $np $bc $lhs.term $rhs.term)
      let (computeTerm?, computeDType?) ← floatingComputeArith? "`-=`" eTerm lhs rhs
      let e' : EOut :=
        { term := eTerm, dtype := lhs.dtype, shape := outShape,
          computeTerm := computeTerm?, computeDType? := computeDType? }
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (name, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident *= $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let name := i.getId.toString
      unless env.any (fun entry => entry.1 == name) do
        Macro.throwError ("`*=`: unknown identifier `" ++ name ++ "`")
      let (lhsDType, lhsShape) ← lookupEnv env name
      if lhsDType == .ptr || lhsDType == .blockPtr then
        Macro.throwError "`*=`: pointer multiplication is not supported"
      let lhsShapeTerm ← lhsShape.term
      let lhsDTypeTerm ← lhsDType.term
      let lhsTerm ← `(Op.ref $lhsDTypeTerm $lhsShapeTerm $nameLit)
      let lhsComputeDType? := lookupComputeDType? env name
      let lhsComputeTerm? ←
        match lhsComputeDType? with
        | some .fp32 => pure (some (← fp32ComputeExpr lhsTerm))
        | some .real => pure (some (← fp64ComputeExpr lhsTerm))
        | some _ =>
            Macro.throwError
              ("identifier `" ++ name ++ "` has unsupported compute dtype annotation")
        | none => pure none
      let lhs : EOut :=
        { term := lhsTerm, dtype := lhsDType, shape := lhsShape,
          computeTerm := lhsComputeTerm?, computeDType? := lhsComputeDType? }
      let rhs0 ← integralSourceLiteral lhs.dtype e (← expandUpdateOperand i e)
      let rhs ←
        if lhs.dtype == .real && rhs0.dtype == .nat then
          match ← expandLeanAntiquoteAs? .real e with
          | some out => pure out
          | none => pure rhs0
        else
          pure rhs0
      let (lhs, rhs) ← coerceNatIntOperands "`*=`" lhs rhs
      let (lhs, rhs) ← coerceRealArithOperands "`*=`" lhs rhs
      ensureComputeArithComposable "`*=` lhs" lhs
      ensureComputeArithComposable "`*=` rhs" rhs
      unless lhs.dtype == rhs.dtype do
        Macro.throwError "`*=`: dtype mismatch"
      let np ← lhs.dtype.numericProof
      let (bc, outShape) ← broadcastTerm lhs.shape rhs.shape "`*=`"
      let eTerm ← `(Op.mul $np $bc $lhs.term $rhs.term)
      let (computeTerm?, computeDType?) ← floatingComputeArith? "`*=`" eTerm lhs rhs
      let e' : EOut :=
        { term := eTerm, dtype := lhs.dtype, shape := outShape,
          computeTerm := computeTerm?, computeDType? := computeDType? }
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (name, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| $i:ident /= $e:tritonExpr) => do
      let nameLit ← identAsStr i
      let name := i.getId.toString
      unless env.any (fun entry => entry.1 == name) do
        Macro.throwError ("`/=`: unknown identifier `" ++ name ++ "`")
      let (lhsDType, lhsShape) ← lookupEnv env name
      if lhsDType == .ptr || lhsDType == .blockPtr then
        Macro.throwError "`/=`: pointer division is not supported"
      let lhsShapeTerm ← lhsShape.term
      let lhsDTypeTerm ← lhsDType.term
      let lhsTerm ← `(Op.ref $lhsDTypeTerm $lhsShapeTerm $nameLit)
      let lhsComputeDType? := lookupComputeDType? env name
      let lhsComputeTerm? ←
        match lhsComputeDType? with
        | some .fp32 => pure (some (← fp32ComputeExpr lhsTerm))
        | some .real => pure (some (← fp64ComputeExpr lhsTerm))
        | some _ =>
            Macro.throwError
              ("identifier `" ++ name ++ "` has unsupported compute dtype annotation")
        | none => pure none
      let lhs : EOut :=
        { term := lhsTerm, dtype := lhsDType, shape := lhsShape,
          computeTerm := lhsComputeTerm?, computeDType? := lhsComputeDType? }
      let rhs0 ← expandUpdateOperand i e
      let rhs ←
        if lhs.dtype == .real && rhs0.dtype == .nat then
          match ← expandLeanAntiquoteAs? .real e with
          | some out => pure out
          | none => pure rhs0
        else
          pure rhs0
      let (lhs, rhs) ← coerceRealArithOperands "`/=`" lhs rhs
      ensureComputeArithComposable "`/=` lhs" lhs
      ensureComputeArithComposable "`/=` rhs" rhs
      unless lhs.dtype == rhs.dtype do
        Macro.throwError "`/=`: dtype mismatch"
      let np ← lhs.dtype.numericProof
      let (bc, outShape) ← broadcastTerm lhs.shape rhs.shape "`/=`"
      let eTerm ← `(Op.div $np $bc $lhs.term $rhs.term)
      let (computeTerm?, computeDType?) ← floatingComputeArith? "`/=`" eTerm lhs rhs
      let e' : EOut :=
        { term := eTerm, dtype := lhs.dtype, shape := outShape,
          computeTerm := computeTerm?, computeDType? := computeDType? }
      let dt ← e'.dtype.term
      let sh ← e'.shape.term
      let exprTerm ←
        match e'.computeTerm with
        | some ce => pure ce
        | none => `(ComputeExpr.alg $e'.term)
      pure (← `(Stmt.assign $dt $sh $nameLit $e'.term),
        ← `(ComputeStmt.assign $dt $sh $nameLit $exprTerm),
        (name, e'.dtype, e'.shape, e'.computeDType?) :: env,
        e'.computeTerm.isSome)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandStore expandExpr expandStaticPtrExpr env p v kwargs (positionalMask := some mask)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr, $kw0:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) => do
      expandStore expandExpr expandStaticPtrExpr env p v (#[kw0] ++ kwargs)
  | `(tritonStmt| tl.store($p:tritonExpr, $v:tritonExpr)) => do
      expandStore expandExpr expandStaticPtrExpr env p v #[]
  | `(tritonStmt| tl.atomic_add($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      expandAtomicAdd expandExpr expandStaticPtrExpr env p v kwargs
  | `(tritonStmt| tl.atomic_max($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_max" p v
  | `(tritonStmt| tl.atomic_min($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_min" p v
  | `(tritonStmt| tl.atomic_and($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_and" p v
  | `(tritonStmt| tl.atomic_or($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_or" p v
  | `(tritonStmt| tl.atomic_xor($p:tritonExpr, $v:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) =>
      unsupportedAtomic2 "tl.atomic_xor" p v
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_xchg" kwargs
      expandAtomicRMWCore "tl.atomic_xchg" (← `(RMWOp.xchg)) none p v none
  | `(tritonStmt| tl.atomic_cas($p:tritonExpr, $cmp:tritonExpr, $v:tritonExpr $[, $kwargs:tritonMemKwarg]*)) => do
      ensureNoAtomicKwargs "tl.atomic_cas" kwargs
      expandAtomicRMWCore "tl.atomic_cas" (← `(RMWOp.cas)) none p cmp (some v)
  | `(tritonStmt| tl.async_copy($dst:tritonExpr, $src:tritonExpr $[, $_kwargs:tritonMemKwarg]*)) => do
      discard <| expandExpr env dst
      discard <| expandExpr env src
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.async_copy"), env, Bool.true)
  | `(tritonStmt| tl.async_wait()) =>
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.async_wait"), env, Bool.true)
  | `(tritonStmt| tl.debug_barrier()) =>
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.effectMarker "tl.debug_barrier"), env, Bool.true)
  | `(tritonStmt| tl.static_print($_msg:term)) =>
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.false) []), env,
        Bool.false)
  | `(tritonStmt| tl.static_assert($cond:tritonExpr $[, $_msg:str]?)) => do
      discard <| expandExpr env cond
      -- Static admission belongs to the caller's compile-time contract. Keep
      -- the canonical no-runtime-work marker executable by both interpreters;
      -- an arbitrary projected algorithm statement remains unsupported.
      pure (← `(Stmt.ifThen (Op.constBool Bool.false) []),
        ← `(ComputeStmt.ifThen (Op.constBool Bool.false) []), env,
        Bool.false)
  | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range(0, $($stop:term), $($step:term)) { $stmts:tritonStmt* }) => do
      let startExpr ← `(tritonExpr| 0)
      let stopExpr ← `(tritonExpr| $($stop:term))
      if signedBounds startExpr stopExpr then
        return ← expandSignedRange i startExpr stopExpr (← `(tritonExpr| $($step:term))) stmts
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forRange $nameLit 0 $stop $step [$algBody,*]),
        ← `(ComputeStmt.forRange $nameLit 0 $stop $step [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $stmts:tritonStmt* }) => do
      let startExpr ← `(tritonExpr| 0)
      if signedBounds startExpr stop then
        return ← expandSignedRange i startExpr stop (← `(tritonExpr| 1)) stmts
      let nameLit ← identAsStr i
      let stop' ← expandNatExpectedExpr env stop
      ensureShape SInfo.scalar stop'.shape "range stop"
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      let zero ← `(Op.constNat 0)
      let one ← `(Op.constNat 1)
      pure (← `(Stmt.forRangeDyn $nameLit $zero $stop'.term $one [$algBody,*]),
        ← `(ComputeStmt.forRangeDyn $nameLit $zero $stop'.term $one [$computeBody,*]),
        env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($($start:term), $($stop:term), $($step:term)) { $stmts:tritonStmt* }) => do
      let startExpr ← `(tritonExpr| $($start:term))
      let stopExpr ← `(tritonExpr| $($stop:term))
      if signedBounds startExpr stopExpr then
        return ← expandSignedRange i startExpr stopExpr (← `(tritonExpr| $($step:term))) stmts
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forRange $nameLit $start $stop $step [$algBody,*]),
        ← `(ComputeStmt.forRange $nameLit $start $stop $step [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr) { $stmts:tritonStmt* }) => do
      if signedBounds start stop then
        return ← expandSignedRange i start stop (← `(tritonExpr| 1)) stmts
      let nameLit ← identAsStr i
      let start' ← expandNatExpectedExpr env start
      let stop' ← expandNatExpectedExpr env stop
      ensureShape SInfo.scalar start'.shape "range start"
      ensureShape SInfo.scalar stop'.shape "range stop"
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      let one ← `(Op.constNat 1)
      pure (← `(Stmt.forRangeDyn $nameLit $start'.term $stop'.term $one [$algBody,*]),
        ← `(ComputeStmt.forRangeDyn $nameLit $start'.term $stop'.term $one [$computeBody,*]),
        env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, -$stop:tritonExpr, -$step:tritonExpr) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let name := i.getId.toString
      let revName := "__rev_" ++ name
      let revNameLit : TSyntax `term := Syntax.mkStrLit revName
      let start' ← expandIntExpectedExpr env start
      let stop' ← expandIntExpectedExpr env stop
      let step' ← expandNatExpectedExpr env step
      ensureShape SInfo.scalar start'.shape "reverse range start"
      ensureShape SInfo.scalar stop'.shape "reverse range stop"
      ensureShape SInfo.scalar step'.shape "reverse range step"
      let bodyEnv :=
        (name, DInfo.int, SInfo.scalar, none) ::
        (revName, DInfo.int, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ←
        expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      -- Mirror the signed interval: j = -i runs from -start to -stop.
      -- The interpreter captures all bounds once, including for an empty loop.
      let zero ← `(Op.constInt 0)
      let lower ← `(Op.sub NumericDType.int Broadcast.nil $zero $start'.term)
      let revRef ← `(Op.ref TileDType.int [] $revNameLit)
      let idxTerm ← `(Op.sub NumericDType.int Broadcast.nil $zero $revRef)
      let idxAssign ← `(Stmt.assign TileDType.int [] $nameLit $idxTerm)
      let computeIdxAssign ←
        `(ComputeStmt.assign TileDType.int [] $nameLit (ComputeExpr.alg $idxTerm))
      pure (← `(Stmt.forRangeInt $revNameLit $lower $stop'.term $step'.term [$idxAssign, $algBody,*]),
        ← `(ComputeStmt.forRangeInt $revNameLit $lower $stop'.term $step'.term
              [$computeIdxAssign, $computeBody,*]),
        env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $step:tritonExpr) { $stmts:tritonStmt* }) => do
      if signedBounds start stop then
        return ← expandSignedRange i start stop step stmts
      let nameLit ← identAsStr i
      let start' ← expandNatExpectedExpr env start
      let stop' ← expandNatExpectedExpr env stop
      let step' ← expandNatExpectedExpr env step
      ensureShape SInfo.scalar start'.shape "range start"
      ensureShape SInfo.scalar stop'.shape "range stop"
      ensureShape SInfo.scalar step'.shape "range step"
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forRangeDyn $nameLit $start'.term $stop'.term $step'.term [$algBody,*]),
        ← `(ComputeStmt.forRangeDyn $nameLit $start'.term $stop'.term $step'.term [$computeBody,*]),
        env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in tl.static_range($($n:term)) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| for $i:ident in tl.static_range($n:num) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.static_range $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let nameLit ← identAsStr i
      let bodyEnv := (i.getId.toString, DInfo.nat, SInfo.scalar, none) :: env
      let (algBody, computeBody, _, bodyHasCompute) ← expandStmts bodyEnv pinned intPinned regionDTypes ptrElems fpSource
        (fpLoopProvenance fpSource.regions fpPtrElems i stmts.toList) facts stmts.toList
      pure (← `(Stmt.forLoop $nameLit $n [$algBody,*]),
        ← `(ComputeStmt.forLoop $nameLit $n [$computeBody,*]), env, bodyHasCompute)
  | `(tritonStmt| tl.if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let thenBaseEnv := env
      let (algThen, computeThen, thenEnv, thenHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.true) thenStmts.toList
      let (algElse, computeElse, elseEnv, elseHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.false) elseStmts.toList
      let nextEnv :=
        if staticBoolCond? env cond then
          mergeStaticBranchEnv thenBaseEnv thenEnv elseEnv
        else
          mergeBranchEnv thenBaseEnv thenEnv elseEnv
      pure (← `(Stmt.ifThenElse $cond'.term [$algThen,*] [$algElse,*]),
        ← `(ComputeStmt.ifThenElse $cond'.term [$computeThen,*] [$computeElse,*]),
        nextEnv,
        cond'.computeTerm.isSome || thenHasCompute || elseHasCompute)
  | `(tritonStmt| tl.if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "tl.if condition"
      ensureShape SInfo.scalar cond'.shape "tl.if condition"
      let (algBody, computeBody, bodyEnv, bodyHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.true) stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$algBody,*]),
        ← `(ComputeStmt.ifThen $cond'.term [$computeBody,*]),
        bodyEnv,
        bodyHasCompute)
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* } else { $se:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } else { $se:tritonStmt* } }))
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* } elif $c2:tritonExpr { $s2:tritonStmt* } else { $se:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } else { if $c2:tritonExpr { $s2:tritonStmt* } else { $se:tritonStmt* } } }))
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* } elif $c2:tritonExpr { $s2:tritonStmt* } elif $c3:tritonExpr { $s3:tritonStmt* } else { $se:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } else { if $c2:tritonExpr { $s2:tritonStmt* } else { if $c3:tritonExpr { $s3:tritonStmt* } else { $se:tritonStmt* } } } }))
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } }))
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* } elif $c2:tritonExpr { $s2:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } else { if $c2:tritonExpr { $s2:tritonStmt* } } }))
  | `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } elif $c1:tritonExpr { $s1:tritonStmt* } elif $c2:tritonExpr { $s2:tritonStmt* } elif $c3:tritonExpr { $s3:tritonStmt* }) =>
      expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts
        (← `(tritonStmt| if $c0:tritonExpr { $s0:tritonStmt* } else { if $c1:tritonExpr { $s1:tritonStmt* } else { if $c2:tritonExpr { $s2:tritonStmt* } else { if $c3:tritonExpr { $s3:tritonStmt* } } } }))
  | `(tritonStmt| if $cond:tritonExpr { $thenStmts:tritonStmt* } else { $elseStmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "if condition"
      ensureShape SInfo.scalar cond'.shape "if condition"
      let (algThen, computeThen, thenEnv, thenHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.true) thenStmts.toList
      let (algElse, computeElse, elseEnv, elseHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.false) elseStmts.toList
      pure (← `(Stmt.ifThenElse $cond'.term [$algThen,*] [$algElse,*]),
        ← `(ComputeStmt.ifThenElse $cond'.term [$computeThen,*] [$computeElse,*]),
        (if staticBoolCond? env cond then
          mergeStaticBranchEnv env thenEnv elseEnv
        else
          mergeBranchEnv env thenEnv elseEnv),
        cond'.computeTerm.isSome || thenHasCompute || elseHasCompute)
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      let cond' ← expandBoolCondition env cond
      ensureDType .bool cond'.dtype "if condition"
      ensureShape SInfo.scalar cond'.shape "if condition"
      let (algBody, computeBody, bodyEnv, bodyHasCompute) ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems
        (withStaticBool env facts cond Bool.true) stmts.toList
      pure (← `(Stmt.ifThen $cond'.term [$algBody,*]),
        ← `(ComputeStmt.ifThen $cond'.term [$computeBody,*]),
        bodyEnv,
        bodyHasCompute)
  | `(tritonStmt| return) =>
      Macro.throwError "`return` is only supported as the sole body statement of `if cond { return }`"
  | _ => Macro.throwUnsupported

private partial def isReturnStmt (st : TSyntax `tritonStmt) : MacroM Bool := do
  pure (st.raw.getKind == ``tritonReturn)

private partial def ifReturnCond? (st : TSyntax `tritonStmt) :
    MacroM (Option (TSyntax `tritonExpr)) := do
  match st with
  | `(tritonStmt| if $cond:tritonExpr { $stmts:tritonStmt* }) => do
      match stmts.toList with
      | [st] =>
          if ← isReturnStmt st then
            pure (some cond)
          else
            pure none
      | _ => pure none
  | _ => pure none

private partial def staticIfParts? (env : Env) (st : TSyntax `tritonStmt) :
    Option (TSyntax `tritonExpr × List (TSyntax `tritonStmt) × List (TSyntax `tritonStmt)) :=
  let parts := match st with
    | `(tritonStmt| if $cond:tritonExpr { $ts:tritonStmt* } else { $es:tritonStmt* }) =>
        some (cond, ts.toList, es.toList)
    | `(tritonStmt| tl.if $cond:tritonExpr { $ts:tritonStmt* } else { $es:tritonStmt* }) =>
        some (cond, ts.toList, es.toList)
    | `(tritonStmt| if $cond:tritonExpr { $ts:tritonStmt* }) =>
        some (cond, ts.toList, [])
    | `(tritonStmt| tl.if $cond:tritonExpr { $ts:tritonStmt* }) =>
        some (cond, ts.toList, [])
    | _ => none
  parts.filter (fun (cond, _, _) => staticBoolCond? env cond)

partial def expandStmts (env : Env) (pinned intPinned : List String)
    (regionDTypes : Inference.RegionDTypes) (ptrElems : Inference.PtrElems)
    (fpSource : FPSourceContext) (fpPtrElems : FPPtrProvenance)
    (facts : StaticBoolFacts) (stmts : List (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `term) × Array (TSyntax `term) × Env × Bool) := do
  match stmts with
  | [] => pure (#[], #[], env, Bool.false)
  | st :: rest => do
      if Inference.isRegionDirective st then
        return ← expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts rest
      if let `(tritonStmt| tl.param $name:ident : $dt:tritonDType (carrier=$carrier:tritonDType)) := st then
        let key := name.getId.toString
        if env.any (fun binding => binding.1 == key) then
          Macro.throwError ("tl.param: register already bound: " ++ key)
        unless (← expandDType dt) == .fp32 && (← expandDType carrier) == .fp32 do
          Macro.throwError "tl.param: explicit carrier requires FP32 precision and FP32 carrier"
        let nextEnv := (key, DInfo.fp32Argument, SInfo.scalar, some DInfo.fp32) :: env
        return ← expandStmts nextEnv pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts rest
      if let `(tritonStmt| tl.param $name:ident : $dt:tritonDType) := st then
        let key := name.getId.toString
        if env.any (fun binding => binding.1 == key) then
          Macro.throwError ("tl.param: register already bound: " ++ key)
        let binding ← match dt with
          | `(tritonDType| tl.int64) =>
              pure (key, DInfo.int, SInfo.scalar, none)
          | _ => do
              let precision ← expandDType dt
              unless precision == .fp32 || precision == .real do
                Macro.throwError "tl.param: only FP32, FP64 and Int64 scalar arguments are supported"
              pure (key, DInfo.real, SInfo.scalar, some precision)
        let nextEnv := binding :: env
        return ← expandStmts nextEnv pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts rest
      match ← ifReturnCond? st with
      | some cond => do
          if let some value := knownStaticBool? env facts cond then
            if value then
              return (#[], #[], env, Bool.false)
            else
              return ← expandStmts env pinned intPinned regionDTypes ptrElems
                fpSource fpPtrElems facts rest
          let cond : TSyntax `tritonExpr :=
            ⟨← annotateMachineLoadCasts regionDTypes ptrElems cond.raw⟩
          let cond' ← expandBoolCondition env cond
          ensureDType .bool cond'.dtype "return condition"
          ensureShape SInfo.scalar cond'.shape "return condition"
          let (algRest, computeRest, restEnv, restHasCompute) ←
            expandStmts env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts rest
          let negCond ← `(Op.boolNot $cond'.term)
          pure (#[
              ← `(Stmt.ifThen $negCond [$algRest,*])
            ], #[
              ← `(ComputeStmt.ifThen $negCond [$computeRest,*])
            ], restEnv, cond'.computeTerm.isSome || restHasCompute)
      | none => do
          -- A repeated constexpr test inside an already selected path has
          -- only one reachable arm. Expand that arm before dtype checking:
          -- the other arm may mention bindings that do not exist on this path.
          if let some (cond, thenStmts, elseStmts) := staticIfParts? env st then
            if let some value := knownStaticBool? env facts cond then
              return ← expandStmts env pinned intPinned regionDTypes ptrElems
                fpSource fpPtrElems facts ((if value then thenStmts else elseStmts) ++ rest)
          -- A constexpr cast may change the register carrier or arithmetic
          -- precision on just one path. Expand the remaining source under
          -- each path's bindings instead of reading the true-arm register on
          -- both paths. Keep the original condition and every source action.
          -- Nested branches retain their rebinding history for the same check.
          if !rest.isEmpty then
            if let some (cond, thenStmts, elseStmts) := staticIfParts? env st then
              let (_, _, thenEnv, _) ← expandStmts env pinned intPinned regionDTypes
                ptrElems fpSource fpPtrElems (withStaticBool env facts cond Bool.true) thenStmts
              let (_, _, elseEnv, _) ← expandStmts env pinned intPinned regionDTypes
                ptrElems fpSource fpPtrElems (withStaticBool env facts cond Bool.false) elseStmts
              if staticBranchChangesType env
                  (staticBranchExitEnv env thenEnv thenStmts)
                  (staticBranchExitEnv env elseEnv elseStmts) then
                let thenTail := (thenStmts ++ rest).toArray
                let elseTail := (elseStmts ++ rest).toArray
                let branched ← `(tritonStmt| if $cond { $thenTail* } else { $elseTail* })
                let (alg, compute, nextEnv, hasCompute) ← expandStmt env pinned intPinned
                  regionDTypes ptrElems fpSource fpPtrElems facts branched
                return (#[alg], #[compute], nextEnv, hasCompute)
          let (algTerm, computeTerm, nextEnv, stmtHasCompute) ←
            expandStmt env pinned intPinned regionDTypes ptrElems fpSource fpPtrElems facts st
          -- Thread the pointer element-dtype map the same way `env` is threaded:
          -- prepend this statement's bindings so the *rest* of the block resolves
          -- a pointer name to its most recent binding, not the body's last one.
          let nextPtrElems := Inference.ptrElemsAfterStmt regionDTypes ptrElems st
          -- Storage evidence after the final statement is never consumed.
          -- Enclosing control-flow continuations analyze their full source
          -- block separately, so their branch-conflict checks remain intact.
          let nextFPPtrElems ← if rest.isEmpty then pure fpPtrElems
            else fpProvenanceAfterStmt fpSource.regions fpPtrElems st
          let (algRest, computeRest, restEnv, restHasCompute) ←
            expandStmts nextEnv pinned intPinned regionDTypes nextPtrElems fpSource nextFPPtrElems facts rest
          pure (#[algTerm] ++ algRest, #[computeTerm] ++ computeRest,
            restEnv, stmtHasCompute || restHasCompute)

end

/-! ## Block elaborator -/

private def dInfoOfTileDTypeExpr? (e : Expr) : MetaM (Option DInfo) := do
  let e ← whnf e
  if e.isConstOf ``TileDType.real then
    pure (some .real)
  else if e.isConstOf ``TileDType.fp32 then
    pure (some .fp32)
  else if e.isConstOf ``TileDType.fp16 then
    pure (some .fp16)
  else if e.isConstOf ``TileDType.bf16 then
    pure (some .bf16)
  else if e.isConstOf ``TileDType.f8e4 then
    pure (some .f8e4)
  else if e.isConstOf ``TileDType.f8e5 then
    pure (some .f8e5)
  else if e.isConstOf ``TileDType.int then
    pure (some .int)
  else if e.isConstOf ``TileDType.nat then
    pure (some .nat)
  else if e.isConstOf ``TileDType.bool then
    pure (some .bool)
  else
    pure none

private def regionDTypeOfLocalType? (type : Expr) : MetaM (Option DInfo) := do
  let type ← whnf type
  match type with
  | Expr.app fn d =>
      if fn.isConstOf ``Region then
        dInfoOfTileDTypeExpr? d
      else
        pure none
  | _ => pure none

private def collectFiniteSourceExternalNames : TermElabM (List String) := do
  let mut names : List String := []
  for decl in (← getLCtx) do
    unless decl.isImplementationDetail do
      names := names ++ [decl.userName.eraseMacroScopes.toString]
  return names

private def collectLocalRegionDTypes : TermElabM Inference.RegionDTypes := do
  let mut out : Inference.RegionDTypes := []
  for decl in (← getLCtx) do
    unless decl.isImplementationDetail do
      match ← regionDTypeOfLocalType? decl.type with
      | some dtype => out := out ++ [(decl.userName.toString, dtype)]
      | none => pure ()
  pure out

private def localNatFVarName? (e : Expr) : TermElabM (Option String) := do
  let e ← whnf e
  match e with
  | Expr.fvar fvarId =>
      match (← getLCtx).find? fvarId with
      | some decl => pure (some decl.userName.toString)
      | none => pure none
  | _ => pure none

private def natEqAliasOfLocalType? (type : Expr) :
    TermElabM (Option (String × String)) := do
  let type ← whnf type
  let args := type.getAppArgs
  if type.getAppFn.isConstOf ``Eq && args.size == 3 then
    let α ← whnf args[0]!
    if α.isConstOf ``Nat then
      match (← localNatFVarName? args[1]!), (← localNatFVarName? args[2]!) with
      | some lhs, some rhs => pure (some (lhs, rhs))
      | _, _ => pure none
    else
      pure none
  else
    pure none

private def collectLocalNatEqAliases : TermElabM (List (String × String)) := do
  let mut out : List (String × String) := []
  for decl in (← getLCtx) do
    unless decl.isImplementationDetail do
      match ← natEqAliasOfLocalType? decl.type with
      | some pair => out := out ++ [pair]
      | none => pure ()
  pure out

partial def rewriteIdentAliases (aliases : List (String × String)) (stx : Syntax) :
    Syntax :=
  if stx.isIdent then
    let key := stx.getId.eraseMacroScopes.toString
    match aliases.find? (fun pair => pair.1 == key) with
    | some (_, target) => mkIdentFrom stx (Name.mkSimple target)
    | none => stx
  else
    match stx with
    | Syntax.node info kind args =>
        Syntax.node info kind (args.map (rewriteIdentAliases aliases))
    | other => other

elab_rules : term
  | `(triton { $stmts:tritonStmt* }) => do
      -- Pre-pass: scan the body for identifiers that appear in `.nat`-pinning
      -- positions (offsets of static-ptr-add chains). The `expandStmt` cases
      -- for `name = tl.load(p)` / `name := tl.load(p)` consult this list
      -- when no explicit `dtype=` kwarg is given, defaulting to `.nat`
      -- instead of `.real` so that `name` participates correctly in pointer
      -- arithmetic downstream.
      let localRegionDTypes ← collectLocalRegionDTypes
      let natEqAliases ← collectLocalNatEqAliases
      let finiteSourceExternalNames ← collectFiniteSourceExternalNames
      let expanded ← liftMacroM do
        let sourceStmts ← stmts.toList.mapM fun st => do
          return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
        let mut explicitPorts : Option (Array (TSyntax `term) × Array (TSyntax `term)) := none
        let mut sourceBody : List (TSyntax `tritonStmt) := []
        for st in sourceStmts do
          match st with
          | `(tritonStmt| tl.ports inputs=[$ins:term,*], outputs=[$outs:term,*]) =>
              if explicitPorts.isSome then
                Macro.throwError "triton: only one top-level tl.ports declaration is allowed"
              explicitPorts := some (ins, outs)
          | _ => sourceBody := sourceBody ++ [st]
        -- Validate the original declaration names before equality aliases can
        -- rewrite syntax. Source helper metadata has its own lexical scope.
        let finiteStmts ← FiniteHelperSource.prepare sourceBody finiteSourceExternalNames
        let stmtList : List (TSyntax `tritonStmt) :=
          finiteStmts.map fun st => ⟨rewriteIdentAliases natEqAliases st.raw⟩
        let stmtList ← RandomProfile.prepare stmtList
        let pinned := Inference.collectNatPinned stmtList
        let intPinned := Inference.collectIntPinned stmtList
        -- Pre-pass: collect `region <name> = <dtype>` directives so the
        -- macro can default `tl.load(R + offs)` / `tl.store(R + offs, _)`
        -- to the declared element dtype instead of `.real`. Local Lean
        -- parameters typed as `Region d` are available here through the term
        -- elaborator and act as implicit region declarations.
        let regionDTypes :=
          Inference.collectRegionDTypes stmtList ++ localRegionDTypes
        -- Pre-pass: propagate region dtypes to chained pointer bindings
        -- (`p = R + offs; ... ; tl.load(p)` recovers `R`'s dtype on the load).
        let ptrElems := Inference.collectPtrElems regionDTypes stmtList
        let fpRegions := Inference.collectRegionDTypes stmtList ++
          localRegionDTypes.filter (fun entry => entry.2 != DInfo.real)
        let fpSource : FPSourceContext := {
          regions := fpRegions
          legacyPointerCasts := fpRegions.isEmpty &&
            !(sourceBody.any (fun st => Inference.hasConcreteRegionStorage st.raw ||
              st.raw.getKind == ``tritonSourceFiniteAlias ||
              st.raw.getKind == ``tritonSourceFiniteHelper)) }
        -- Evidence starts at the source ports and follows execution order;
        -- later assignments must not establish provenance for earlier loads.
        let fpPtrElems : FPPtrProvenance := []
        let (_, computeStmtTerms, _, _) ←
          expandStmts [] pinned intPinned regionDTypes ptrElems fpSource fpPtrElems [] stmtList
        -- Auto-scan body: collect every region appearing in `tl.load(...)` (inputs)
        -- and `tl.store(...)` (outputs). Order = body occurrence; no macro-time
        -- dedup (a mix of literals and Lean terms can't be statically deduped, and
        -- `Kernel.inputs/outputs` is metadata-only, so duplicates are harmless).
        let (insArr, outsArr) := match explicitPorts with
          | some ports => ports
          | none => let (ins, outs) := Metadata.blockRegions sourceBody
                    (ins.toArray, outs.toArray)
        let result ← `(ComputeKernel.mk [$insArr,*] [$outsArr,*] [$computeStmtTerms,*])
        normalizePrecisionScopes none result.raw
      let e ← elabTerm expanded none
      -- Resolve generated placeholders before declaration processing traverses
      -- the expanded expression. Large kernels otherwise repeatedly expand
      -- shared metavariable assignments, causing excessive memory use.
      synthesizeSyntheticMVarsNoPostponing
      instantiateMVars e

end VeriTile.Triton.DSL
