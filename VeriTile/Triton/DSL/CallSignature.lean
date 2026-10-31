import VeriTile.Triton.DSL.SourceDType
import VeriTile.Triton.DSL.RawBodyControl
import VeriTile.Triton.DSL.RawSourceText
import VeriTile.Triton.DSL.RawBodyConstants
import VeriTile.Triton.DSL.PositionalBlockPointer
import VeriTile.Triton.DSL.RawBodyStores
import VeriTile.Triton.DSL.RawBodyDTypeControl
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
  /-- Floating storage supplied by a polymorphic call site. Unlike the fixed
  source-dtype entries this may depend on the caller's FloatDType parameters. -/
  floatingPointers : List (String × FloatDType) := []
  scalars : List (String × ScalarType) := []
  constexpr : List String := []

namespace DSL.CallSignature
open Lean Elab Term Meta

syntax "tl.zeros(" "(" tritonExpr,* ")" ")" : tritonExpr
syntax "tl.zeros(" "(" tritonExpr,* ")" ", " tritonDType ")" : tritonExpr
syntax "tl.zeros(" "(" tritonExpr,* ")" ", " ident "=" tritonDType ")" : tritonExpr
syntax "for " ident " in " "range(0, " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax (priority := low) "for " ident " in " "range(0, " tritonExpr ", " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
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

private partial def normalizeRawBody (parserEnv : Environment) (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM (normalizeRawBody parserEnv))
  match raw with
  | `(tritonExpr| tl.cast($value:tritonExpr, $dtype:tritonDType)) => do
      unless integerDType dtype do return raw
      let some n := pythonIntegerLiteral? value | return raw
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
  | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr, $step:tritonExpr) { $body:tritonStmt* }) =>
      return (← `(tritonStmt| for $i:ident in range($(0), $stop:tritonExpr, $step:tritonExpr) { $body* })).raw
  | _ => return raw

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

@[term_elab withSignature] def elaborateSignature : TermElab := fun stx expected => do
      let profile : TSyntax `term := ⟨stx[1]⟩
      let raw := stx.getArgs.back!
      let source ← match RawSourceText.normalize raw.getAtomVal with
        | .ok body => pure body
        | .error message => throwErrorAt raw "{message}"
      let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
        | .ok body => pure body
        | .error message => throwErrorAt raw "{message}"
      let `(triton { $stmts:tritonStmt* }) := parsed
        | throwErrorAt raw "expected a complete Triton function body"
      let profile ← elabTermEnsuringType profile (mkConst ``Triton.CallSignature)
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
      let names := pointers.map Prod.fst ++ floatingPointers.map Prod.fst ++ scalars.map Prod.fst ++ constants
      unless names.eraseDups.length == names.length do throwError "duplicate name in call signature"
      for stmt in stmts do rejectBodyMetadata stmt.raw
      let (constantBindings,prepared) ← liftMacroM <|
        RawBodyConstants.prepare (constants.map Name.mkSimple) [] stmts.toList
      let stmts := prepared.toArray
      let parserEnv ← getEnv
      let stmts ← liftMacroM <| stmts.mapM fun stmt =>
        return (⟨← normalizeRawBody parserEnv (← PositionalBlockPointer.normalize (← RawBodyControl.normalize stmt.raw))⟩ : TSyntax `tritonStmt)
      let (reads, writes) := Metadata.blockRegions stmts.toList
      for region in reads ++ writes do
        let `(term| $id:ident) := region
          | throwErrorAt region "call signature requires named external pointer parameters"
        unless (pointers.map Prod.fst ++ floatingPointers.map Prod.fst).contains id.getId.eraseMacroScopes.toString do
          throwErrorAt region "pointer parameter {id} is missing from the call signature"
      let term ← liftMacroM do
        let mut declarations : Array (TSyntax `tritonStmt) := #[]
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
            | _ => pure none
          if let some dtypeTerm := dtypeTerm then dtypeBindings := dtypeBindings ++ [(name,dtypeTerm)]
        for (name, dtype) in floatingPointers do
          let id := mkIdent (Name.mkSimple name)
          declarations := declarations.push (← `(tritonStmt| tl.region $id:ident = storage_dtype($dtype:ident)))
          dtypeBindings := dtypeBindings ++ [(name,⟨dtype.raw⟩)]
        for (name, kind) in scalars do
          let id := mkIdent (Name.mkSimple name)
          let dtName := match kind with
            | .fp32 => `tl.float32 | .fp64 => `tl.float64 | .integer => `tl.int64
          let some dt ← dtypeSyntaxOfName? dtName
            | Macro.throwError "unsupported scalar dtype"
          declarations := declarations.push (← `(tritonStmt| tl.param $id:ident : $dt:tritonDType))
        RawBodyDTypeControl.emit dtypeBindings (fun selected => do
          let mut body := selected
          for (name, kind) in pointers do
            body ← RawBodyStores.lower (SourceDType.storeValue kind) [Name.mkSimple name] body
          for (name, dtype) in floatingPointers do
            body ← RawBodyStores.lower (fun value =>
              `(tritonExpr| ($value:tritonExpr).to(storage_dtype($dtype:ident)))) [Name.mkSimple name] body
          let originalStores := selected.flatMap (fun st => storeValues st.raw)
          let convertedStores := body.flatMap (fun st => storeValues st.raw)
          unless originalStores.length == convertedStores.length do
            Macro.throwError "implicit store conversion changed the source store count"
          for (original, converted) in originalStores.zip convertedStores do
            if original == converted then
              Macro.throwErrorAt original "cannot determine tl.store destination dtype from the call signature"
          let emitted := declarations ++ body.toArray
          `(triton { $emitted* })) stmts.toList
      let term ← liftMacroM <| constantBindings.reverse.foldlM (init:=term) fun body (name,value) =>
        `(let $name:ident : Nat := $value:term; $body:term)
      let term ← liftMacroM <| storageBindings.reverse.foldlM (init:=term) fun body (name,value) =>
        `(let $name:ident : FloatDType := $value:term; $body:term)
      elabTerm term expected

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
