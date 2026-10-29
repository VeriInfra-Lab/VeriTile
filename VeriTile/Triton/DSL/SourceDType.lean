import VeriTile.Triton.DSL

/-! Source-level dtype specialization. The body retains dtype queries and
constexpr assignments; specialization evaluates only these static expressions.
Runtime loads, casts, arithmetic, branches and stores remain in the quotation. -/
namespace VeriTile.Triton

inductive SourceDType where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5
  | i8 | i16 | i32 | i64 | u8 | u16 | u32 | u64 | boolean
  deriving DecidableEq, Repr

namespace DSL.SourceDType
open Lean

syntax ident " : " "tl.constexpr" " = " tritonExpr : tritonStmt
syntax:40 tritonExpr:41 " or " tritonExpr:41 : tritonExpr
syntax "triton " "(" &"dtype" ident " := " term ")" " {" tritonStmt* "}" : term

inductive Constant where
  | integer (value : Int)
  | boolean (value : Bool)
  | negativeInfinity
  deriving Inhabited

abbrev Bindings := List (Name × Constant)

def dtypeName : Triton.SourceDType → Name
  | .fp16 => `tl.float16 | .bf16 => `tl.bfloat16 | .fp32 => `tl.float32 | .fp64 => `tl.float64
  | .f8e4 => `tl.float8e4nv | .f8e5 => `tl.float8e5
  | .i8 => `tl.int8 | .i16 => `tl.int16 | .i32 => `tl.int32 | .i64 => `tl.int64
  | .u8 => `tl.uint8 | .u16 => `tl.uint16 | .u32 => `tl.uint32 | .u64 => `tl.uint64
  | .boolean => `tl.int1

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

private partial def prepare (pointer : Name) (dtype : Triton.SourceDType)
    (bindings : Bindings) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt) × Bindings) := do
  match body with
  | [] => return ([], bindings)
  | stmt :: tail =>
      match stmt with
      | `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr) =>
          let some c := evaluate pointer dtype bindings value
            | Macro.throwErrorAt value "unsupported dtype constexpr expression"
          prepare pointer dtype ((name.getId,c)::bindings) tail
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          if let some (.boolean c) := evaluate pointer dtype bindings condition then
            return ← prepare pointer dtype bindings ((if c then yes.toList else no.toList) ++ tail)
          for st in yes do rejectDynamicConstants st.raw
          let (yes, _) ← prepare pointer dtype bindings yes.toList
          for st in no do rejectDynamicConstants st.raw
          let (no, _) ← prepare pointer dtype bindings no.toList
          let condition : TSyntax `tritonExpr := ⟨← substitute pointer dtype bindings condition.raw⟩
          let yes := yes.toArray; let no := no.toArray
          let head ← `(tritonStmt| if $condition { $yes* } else { $no* })
          let (tail, next) ← prepare pointer dtype bindings tail
          return (head::tail,next)

      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          if let some (.boolean c) := evaluate pointer dtype bindings condition then
            return ← prepare pointer dtype bindings ((if c then yes.toList else []) ++ tail)
          for st in yes do rejectDynamicConstants st.raw
          let (yes, _) ← prepare pointer dtype bindings yes.toList
          let condition : TSyntax `tritonExpr := ⟨← substitute pointer dtype bindings condition.raw⟩
          let yes := yes.toArray
          let head ← `(tritonStmt| if $condition { $yes* })
          let (tail, next) ← prepare pointer dtype bindings tail
          return (head::tail,next)
      | _ =>
          if let `(tritonStmt| $name:ident = $_value:tritonExpr) := stmt then
            if bindings.any (·.1 == name.getId) then
              Macro.throwErrorAt name "runtime assignment shadows a dtype constexpr"
          rejectDynamicConstants stmt.raw
          let head := (⟨← substitute pointer dtype bindings stmt.raw⟩ : TSyntax `tritonStmt)
          let (tail, next) ← prepare pointer dtype bindings tail
          return (head::tail,next)

private partial def containsStore (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal.startsWith "tl.store(") || raw.getArgs.any containsStore

private partial def pointerExpr (aliases : List Name) (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $name:ident) => aliases.contains name.getId
  | `(tritonExpr| ($inner:tritonExpr)) => pointerExpr aliases inner
  | `(tritonExpr| $base:tritonExpr + $_offset:tritonExpr) => pointerExpr aliases base
  | _ => Bool.false

private partial def rejectBranchAliases (aliases : List Name) (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $name:ident = $value:tritonExpr) := raw then
    if aliases.contains name.getId || pointerExpr aliases value then
      Macro.throwErrorAt raw "dtype pointer aliases cannot be rebound under runtime control flow"
  for arg in raw.getArgs do rejectBranchAliases aliases arg

private def storeValue (dtype : Triton.SourceDType) (value : TSyntax `tritonExpr) :
    MacroM (TSyntax `tritonExpr) := do
  let some dt ← dtypeSyntaxOfName? (dtypeName dtype)
    | Macro.throwError "unsupported implicit store dtype"
  if dtype == .boolean then
    -- The selected bool ABI truncates to i8 before testing the stored byte.
    `(tritonExpr| (($value:tritonExpr).to(tl.int8)) != $( (0 : Int) ))
  else `(tritonExpr| ($value:tritonExpr).to($dt:tritonDType))

/-- Lower the implicit destination conversion of `tl.store` for this pointer.
This pass follows direct pointer aliases and does not change the source body.
It never replaces arithmetic, NaN selection or clamp instructions. -/
partial def lowerStores (dtype : Triton.SourceDType) (aliases : List Name)
    (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | stmt :: tail =>
      let mut next := aliases
      let mut stmt := stmt
      match stmt with
      | `(tritonStmt| $name:ident = $value:tritonExpr) =>
          next := aliases.filter (· != name.getId)
          if pointerExpr aliases value then next := name.getId :: next
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr)) =>
          if pointerExpr aliases destination then
            let converted ← storeValue dtype value
            stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $converted:tritonExpr))
      | `(tritonStmt| tl.store($destination:tritonExpr, $value:tritonExpr,  $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*)) =>
          if pointerExpr aliases destination then
            let converted ← storeValue dtype value
            stmt ← `(tritonStmt| tl.store($destination:tritonExpr, $converted:tritonExpr,  $kwarg:tritonMemKwarg $[, $kwargs:tritonMemKwarg]*))
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
          for st in yes do rejectBranchAliases aliases st.raw
          for st in no do rejectBranchAliases aliases st.raw
          let yes := (← lowerStores dtype aliases yes.toList).toArray
          let no := (← lowerStores dtype aliases no.toList).toArray
          stmt ← `(tritonStmt| if $condition { $yes* } else { $no* })
      | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
          for st in yes do rejectBranchAliases aliases st.raw
          let yes := (← lowerStores dtype aliases yes.toList).toArray
          stmt ← `(tritonStmt| if $condition { $yes* })
      | _ =>
          rejectBranchAliases aliases stmt.raw
          if containsStore stmt.raw then
            Macro.throwErrorAt stmt "unsupported store control flow in dtype specialization"
      return stmt :: (← lowerStores dtype next tail)

/-- Specialize only source dtype queries and their constexpr bindings. Pasted
body callers retain their own checked pointer ABI and destination conversions. -/
def specializeBody (pointer : Name) (dtype : Triton.SourceDType)
    (stmts : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt)) := do
  let normalized ← stmts.toList.mapM fun st =>
    return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
  let (body, _) ← prepare pointer dtype [] normalized
  return body.toArray

macro_rules
  | `(triton (dtype $pointer:ident := $kind:term) { $stmts:tritonStmt* }) => do
      let kinds : List Triton.SourceDType := [.fp16,.bf16,.fp32,.fp64,.f8e4,.f8e5,.i8,.i16,.i32,.i64,.u8,.u16,.u32,.u64,.boolean]
      let constructors := [``Triton.SourceDType.fp16,``Triton.SourceDType.bf16,``Triton.SourceDType.fp32,
        ``Triton.SourceDType.fp64,``Triton.SourceDType.f8e4,``Triton.SourceDType.f8e5,``Triton.SourceDType.i8,``Triton.SourceDType.i16,``Triton.SourceDType.i32,
        ``Triton.SourceDType.i64,``Triton.SourceDType.u8,``Triton.SourceDType.u16,``Triton.SourceDType.u32,
        ``Triton.SourceDType.u64,``Triton.SourceDType.boolean]
      let mut alternatives : Array (TSyntax ``Lean.Parser.Term.matchAlt) := #[]
      for (dtype, constructor) in kinds.zip constructors do
        let normalized ← stmts.toList.mapM fun st =>
          return (⟨← Inference.normalizeControlFlow st.raw⟩ : TSyntax `tritonStmt)
        let (body, _) ← prepare pointer.getId dtype [] normalized
        let body ← lowerStores dtype [pointer.getId] body
        let some dt ← dtypeSyntaxOfName? (dtypeName dtype)
          | Macro.throwError "unsupported source dtype"
        let declaration ← `(tritonStmt| tl.region $pointer:ident = $dt:tritonDType)
        let body := (declaration::body).toArray
        let ctor := mkIdent constructor
        alternatives := alternatives.push (← `(Lean.Parser.Term.matchAltExpr| | $ctor:ident => triton { $body* }))
      `(match ($kind : Triton.SourceDType) with $alternatives:matchAlt*)

end DSL.SourceDType
end VeriTile.Triton
