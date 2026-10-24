import VeriTile.Triton.DSL.SourceDType
import Lean

/-! Call-site typing for a pasted Triton body. The signature describes pointer
storage and runtime scalar registers; it emits no kernel instruction. The body
uses ordinary Triton names, casts and implicit destination conversions. -/
namespace VeriTile.Triton

/-- Scalar carrier/precision supplied by the invocation. `integer` is the
mathematical integer channel, not a claim about an unannotated Python ABI width. -/
inductive ScalarType where
  | fp32 | fp64 | integer

structure CallSignature where
  pointers : List (String × SourceDType) := []
  scalars : List (String × ScalarType) := []
  constexpr : List String := []

namespace DSL.CallSignature
open Lean Elab Term Meta

syntax "with_triton_signature " term " in " "triton" " {" tritonStmt* "}" : term

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
  throwError "runtime scalar type must be FP32, FP64 or integer"

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
  for arg in raw.getArgs do rejectBodyMetadata arg

private partial def storeValues (raw : Syntax) : List Syntax :=
  match raw with
  | `(tritonStmt| tl.store($_:tritonExpr, $value:tritonExpr)) => [value.raw]
  | `(tritonStmt| tl.store($_:tritonExpr, $value:tritonExpr, $_:tritonMemKwarg $[, $_:tritonMemKwarg]*)) => [value.raw]
  | _ => raw.getArgs.toList.flatMap storeValues

elab_rules : term
  | `(with_triton_signature $profile:term in triton { $stmts:tritonStmt* }) => do
      let profile ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
      let pointers ← bindings dtypeValue (← mkAppM ``Triton.CallSignature.pointers #[profile])
      let scalars ← bindings scalarTypeValue (← mkAppM ``Triton.CallSignature.scalars #[profile])
      let constants ← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM stringValue
      let names := pointers.map Prod.fst ++ scalars.map Prod.fst ++ constants
      unless names.eraseDups.length == names.length do throwError "duplicate name in call signature"
      for stmt in stmts do rejectBodyMetadata stmt.raw
      let (reads, writes) := Metadata.blockRegions stmts.toList
      for region in reads ++ writes do
        let `(term| $id:ident) := region
          | throwErrorAt region "call signature requires named external pointer parameters"
        unless pointers.any (fun p => p.1 == id.getId.eraseMacroScopes.toString) do
          throwErrorAt region "pointer parameter {id} is missing from the call signature"
      let term ← liftMacroM do
        let mut declarations : Array (TSyntax `tritonStmt) := #[]
        for (name, kind) in pointers do
          let id := mkIdent (Name.mkSimple name)
          let some dt ← dtypeSyntaxOfName? (SourceDType.dtypeName kind)
            | Macro.throwError "unsupported pointer dtype"
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = $dt:tritonDType))
        for (name, kind) in scalars do
          let id := mkIdent (Name.mkSimple name)
          let dtName := match kind with
            | .fp32 => `tl.float32 | .fp64 => `tl.float64 | .integer => `tl.int64
          let some dt ← dtypeSyntaxOfName? dtName
            | Macro.throwError "unsupported scalar dtype"
          declarations := declarations.push (← `(tritonStmt| tl.param $id:ident : $dt:tritonDType))
        let mut body := stmts.toList
        for (name, kind) in pointers do
          body ← SourceDType.lowerStores kind [Name.mkSimple name] body
        let originalStores := stmts.toList.flatMap (fun st => storeValues st.raw)
        let convertedStores := body.flatMap (fun st => storeValues st.raw)
        unless originalStores.length == convertedStores.length do
          Macro.throwError "implicit store conversion changed the source store count"
        for (original, converted) in originalStores.zip convertedStores do
          if original == converted then
            Macro.throwErrorAt original "cannot determine tl.store destination dtype from the call signature"
        let emitted := declarations ++ body.toArray
        `(triton { $emitted* })
      elabTerm term none

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
