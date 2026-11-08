import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.SourceDType

/-! Resolve immutable pointer dtype properties and branch-local dtype constexprs
using caller metadata before ordinary runtime source lowering. The queried
pointer must remain immutable; multiple queried pointers are refused. This optional
adapter evaluates only the static fragment supported by SourceDType; actual
loads, arithmetic, casts and stores continue through CallSignature. -/
namespace VeriTile.Triton.DSL.PointerDTypeConstants
open Lean Elab Term Meta
open SourceDType (Constant Bindings)
private def bits : Triton.SourceDType → Nat
  | .fp16 | .bf16 | .i16 | .u16 => 16
  | .fp32 | .i32 | .u32 => 32
  | .fp64 | .i64 | .u64 => 64
  | .i8 | .u8 | .f8e4 | .f8e5 => 8
  | .boolean => 1

private def floating : Triton.SourceDType → Bool
  | .fp16 | .bf16 | .fp32 | .fp64 | .f8e4 | .f8e5 => Bool.true
  | _ => Bool.false

private def signed : Triton.SourceDType → Bool
  | .i8 | .i16 | .i32 | .i64 => Bool.true
  | _ => Bool.false

private def property (pointer : Name) (dtype : Triton.SourceDType) (name : Name) : Option Constant := do
  let propertyPrefix := pointer.toString ++ ".dtype.element_ty."
  unless name.toString.startsWith propertyPrefix do none
  match (name.toString.dropPrefix propertyPrefix).toString with
  | "is_int" => some (.boolean (!floating dtype))
  | "is_int1" => some (.boolean (dtype == .boolean))
  | "is_int_signed" => some (.boolean (signed dtype))
  | "is_floating" => some (.boolean (floating dtype))
  | "primitive_bitwidth" => some (.integer (bits dtype))
  | _ => none

private partial def evaluate (pointer : Name) (dtype : Triton.SourceDType)
    (bindings : Bindings) (e : TSyntax `tritonExpr) : Option Constant := do
  match e with
  | `(tritonExpr| -float("inf")) => some .negativeInfinity
  | `(tritonExpr| -(float("inf"))) => some .negativeInfinity
  | `(tritonExpr| float("-inf")) => some .negativeInfinity
  | `(tritonExpr| $n:num) => some (.integer n.getNat)
  | `(tritonExpr| true) => some (.boolean Bool.true)
  | `(tritonExpr| false) => some (.boolean Bool.false)
  | `(tritonExpr| $name:ident) =>
      (bindings.find? (·.1 == name.getId)).map (·.2) <|> property pointer dtype name.getId
  | `(tritonExpr| $name:ident()) => property pointer dtype name.getId
  | `(tritonExpr| ($x:tritonExpr)) => evaluate pointer dtype bindings x
  | `(tritonExpr| -$x:tritonExpr) =>
      let .integer x ← evaluate pointer dtype bindings x | none
      some (.integer (-x))
  | `(tritonExpr| ($x:tritonExpr if $c:tritonExpr else $y:tritonExpr)) =>
      let .boolean c ← evaluate pointer dtype bindings c | none
      evaluate pointer dtype bindings (if c then x else y)
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) =>
      let .boolean a ← evaluate pointer dtype bindings a | none
      let .boolean b ← evaluate pointer dtype bindings b | none
      some (.boolean (a || b))
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) =>
      let .boolean a ← evaluate pointer dtype bindings a | none
      let .boolean b ← evaluate pointer dtype bindings b | none
      some (.boolean (a && b))
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) =>
      let .integer a ← evaluate pointer dtype bindings a | none
      let .integer b ← evaluate pointer dtype bindings b | none
      some (.integer (a-b))
  | `(tritonExpr| $a:tritonExpr << $b:tritonExpr) =>
      let .integer a ← evaluate pointer dtype bindings a | none
      let .integer b ← evaluate pointer dtype bindings b | none
      if b < 0 then none else some (.integer (a * 2^b.toNat))
  | `(tritonExpr| $a:tritonExpr > $b:tritonExpr) =>
      let .integer a ← evaluate pointer dtype bindings a | none
      let .integer b ← evaluate pointer dtype bindings b | none
      some (.boolean (a>b))
  | _ => none

private def literal : Constant → MacroM (TSyntax `tritonExpr)
  | .negativeInfinity => `(tritonExpr| -float("inf"))
  | .boolean Bool.true => `(tritonExpr| true)
  | .boolean Bool.false => `(tritonExpr| false)
  | .integer n => do
      let value : TSyntax `num := ⟨Syntax.mkNumLit (toString n.natAbs)⟩
      if n < 0 then `(tritonExpr| -$value:num) else `(tritonExpr| $value:num)

/-- Replace only constexpr names/properties. Never fold a runtime expression. -/
private partial def substitute (pointer : Name) (dtype : Triton.SourceDType)
    (bindings : Bindings) (raw : Syntax) : MacroM Syntax := do
  if let some c := evaluate pointer dtype bindings ⟨raw⟩ then
    return (← literal c).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute pointer dtype bindings))
  | other => return other

/-- Runtime control flow cannot define a compile-time binding. Reject it instead
of letting the value escape a branch with an arbitrary specialization. -/
private partial def rejectDynamicConstants (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $_name:ident : tl.constexpr = $_value:tritonExpr) := raw then
    Macro.throwErrorAt raw "dtype constexpr assignment under runtime control flow"
  for arg in raw.getArgs do rejectDynamicConstants arg

private partial def specializeBody (pointer : Name) (dtype : Triton.SourceDType)
    (bindings : Bindings) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt) × Bindings) := do
  match body with
  | [] => return ([], bindings)
  | stmt :: tail =>
      match stmt with
      | `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr) =>
          let some c := evaluate pointer dtype bindings value
            | Macro.throwErrorAt value "unsupported dtype constexpr expression"
          specializeBody pointer dtype ((name.getId,c)::bindings) tail
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          if let some (.boolean c) := evaluate pointer dtype bindings condition then
            return ← specializeBody pointer dtype bindings ((if c then yes.toList else no.toList) ++ tail)
          for st in yes do rejectDynamicConstants st.raw
          let (yes, _) ← specializeBody pointer dtype bindings yes.toList
          for st in no do rejectDynamicConstants st.raw
          let (no, _) ← specializeBody pointer dtype bindings no.toList
          let condition : TSyntax `tritonExpr := ⟨← substitute pointer dtype bindings condition.raw⟩
          let yes := yes.toArray; let no := no.toArray
          let head ← `(tritonStmt| if $condition { $yes* } else { $no* })
          let (tail, next) ← specializeBody pointer dtype bindings tail
          return (head::tail,next)

      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          if let some (.boolean c) := evaluate pointer dtype bindings condition then
            return ← specializeBody pointer dtype bindings ((if c then yes.toList else []) ++ tail)
          for st in yes do rejectDynamicConstants st.raw
          let (yes, _) ← specializeBody pointer dtype bindings yes.toList
          let condition : TSyntax `tritonExpr := ⟨← substitute pointer dtype bindings condition.raw⟩
          let yes := yes.toArray
          let head ← `(tritonStmt| if $condition { $yes* })
          let (tail, next) ← specializeBody pointer dtype bindings tail
          return (head::tail,next)
      | _ =>
          if let `(tritonStmt| $name:ident = $_value:tritonExpr) := stmt then
            if bindings.any (·.1 == name.getId) then
              Macro.throwErrorAt name "runtime assignment shadows a dtype constexpr"
          rejectDynamicConstants stmt.raw
          let head := (⟨← substitute pointer dtype bindings stmt.raw⟩ : TSyntax `tritonStmt)
          let (tail, next) ← specializeBody pointer dtype bindings tail
          return (head::tail,next)


private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "pointer dtype metadata must be a statically known list"

private partial def usedPointer (raw : Syntax) (name : String) : Bool :=
  (raw.isIdent && raw.getId.eraseMacroScopes.toString.startsWith (name++".dtype.element_ty.")) ||
    raw.getArgs.any (fun child=>usedPointer child name)

private partial def immutablePointer (pointer : Name) (raw : Syntax) : MacroM Unit := do
  let args:=raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i:=args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun a=>a.isIdent && a.getId.eraseMacroScopes==pointer.eraseMacroScopes) then
        Macro.throwError "a queried pointer dtype requires an immutable pointer parameter"
  if let a::b::_:=args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        b.isIdent && b.getId.eraseMacroScopes==pointer.eraseMacroScopes then
      Macro.throwError "a loop cannot shadow a queried pointer parameter"
  for child in raw.getArgs do immutablePointer pointer child

private partial def noQueries (pointer : Name) (raw : Syntax) : MacroM Unit := do
  if usedPointer raw pointer.toString then
    if raw.isIdent then Macro.throwError "unsupported pointer dtype constexpr property"
    for child in raw.getArgs do noQueries pointer child

private def elaborate (stx : Syntax) (expected : Option Expr) (withScalars : Bool) : TermElabM Expr := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body => pure body | .error message => throwErrorAt raw "{message}"
  let entries ← listValues (← mkAppM ``CallSignature.pointers #[profile])
  let mut queried : List (Name × Expr) := []
  for pair in entries do
    let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair]) | throwError "pointer name must be literal"
    if usedPointer parsed name then queried:=queried++[(name.toName,← whnf (← mkAppM ``Prod.snd #[pair]))]
  if queried.isEmpty then
    return ← if withScalars then CallSignature.elaborateInvocation stx expected
      else CallSignature.elaborateSignature stx expected
  unless queried.length==1 do throwError "this dtype-constant fragment requires one queried pointer"
  let (pointer,value):=queried.head!
  liftMacroM (immutablePointer pointer parsed)
  let kinds : List Triton.SourceDType := [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean]
  let ctors := [``Triton.SourceDType.fp16,``Triton.SourceDType.bf16,``Triton.SourceDType.fp32,``Triton.SourceDType.fp64,``Triton.SourceDType.f8e4,``Triton.SourceDType.f8e5,``Triton.SourceDType.i8,``Triton.SourceDType.i16,``Triton.SourceDType.i32,``Triton.SourceDType.i64,``Triton.SourceDType.u8,``Triton.SourceDType.u16,``Triton.SourceDType.u32,``Triton.SourceDType.u64,``Triton.SourceDType.boolean]
  let branch (dtype : Triton.SourceDType) (selected : Expr) : TermElabM Expr := do
    let `(triton { $stmts:tritonStmt* }):=parsed | throwError "expected a complete source body"
    let normalized ← liftMacroM <| stmts.toList.mapM fun st =>
      return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
    let (body,_) ← liftMacroM (specializeBody pointer dtype [] normalized)
    for stmt in body do liftMacroM (noQueries pointer stmt.raw)
    let selected ← exprToSyntax selected
    let body:=body.toArray
    -- Reuse the generic source renderer after compile-time specialization.
    let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
    let stx:=stx.setArgs ((stx.getArgs.set! 1 selected.raw).set! (stx.getArgs.size-1) raw)
    if withScalars then CallSignature.elaborateInvocation stx expected else CallSignature.elaborateSignature stx expected
  if let some (dtype,_):=(kinds.zip ctors).find? (fun (_,ctor)=>value.isConstOf ctor) then
    return ← branch dtype profile
  let branches ← (kinds.zip ctors).mapM fun (dtype,ctor) =>
    branch dtype (profile.replace fun e=>if e==value then some (mkConst ctor) else none)
  let motive:=mkLambda `dtype BinderInfo.default (mkConst ``Triton.SourceDType) (mkConst ``ComputeKernel)
  mkAppOptM ``Triton.SourceDType.casesOn (#[some motive,some value]++branches.toArray.map some)

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => elaborate stx expected Bool.false
@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => elaborate stx expected Bool.true
end VeriTile.Triton.DSL.PointerDTypeConstants
