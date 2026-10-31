import VeriTile.Triton.DSL
import Lean

/-! Proof-only views of natural-index expressions in a named source.

The scalar view folds products of Nat constants and associates a trailing scalar
factor. The range view additionally lowers zero-based loops to unit-step ranges
and `(Nat.toInt x - Nat.toInt c).toNat` to natural subtraction. Floating
expressions, casts and effects remain in the source-derived tree.

These elaborators do not prove semantic equivalence and must not construct a
canonical kernel. Callers must prove the complete connection, including failure
and memory behavior. DiagSsmTriton's ForwardProjection and BackwardProjection
check regional and flat execution of these derived views for all parameters and
initial states; the recurrence proofs then use their checked decompositions.
-/
namespace VeriTile.Meta
open Lean Lean.Meta Lean.Elab Lean.Elab.Term
open VeriTile.Triton
syntax "source_nat_view " term : term
syntax "source_nat_range_view " term : term

private partial def intNatConstant? (e : Expr) : Option Expr := Id.run do
  if e.isAppOfArity ``Op.constInt 1 then
    let value := e.getAppArgs[0]!
    if value.isAppOfArity ``Int.ofNat 1 then return some value.getAppArgs[0]!
    else return none
  if e.isAppOf ``Op.mul then
    let a := e.getAppArgs
    let some left := intNatConstant? a[a.size-2]! | return none
    let some right := intNatConstant? a[a.size-1]! | return none
    return some (mkApp2 (Lean.mkConst ``Nat.mul) left right)
  return none

private def normalizeBackward (e : Expr) : Option Expr := Id.run do
  if e.isAppOf ``ComputeStmt.forLoop then
    let args := e.getAppArgs
    return some (mkAppN (Lean.mkConst ``ComputeStmt.forRange)
      #[args[0]!, mkNatLit 0, args[1]!, mkNatLit 1, args[2]!])
  if e.isAppOf ``Stmt.forLoop then
    let args := e.getAppArgs
    return some (mkAppN (Lean.mkConst ``Stmt.forRange)
      #[args[0]!, mkNatLit 0, args[1]!, mkNatLit 1, args[2]!])
  unless e.isAppOf ``Op.castIntToNat do return none
  let sub := e.getAppArgs.back!
  unless sub.isAppOf ``Op.sub do return none
  let args := sub.getAppArgs
  let left := args[args.size-2]!
  let right := args[args.size-1]!
  unless left.isAppOf ``Op.castNatToInt do return none
  let some n := intNatConstant? right | return none
  let next := args.map (fun a => if a.isConstOf ``VeriTile.Triton.NumericDType.int then Lean.mkConst ``VeriTile.Triton.NumericDType.nat else if a.isConstOf ``TileDType.int then Lean.mkConst ``TileDType.nat else a)
  let next := next.set! (args.size-2) left.getAppArgs.back!
    |>.set! (args.size-1) (mkApp (Lean.mkConst ``Op.constNat) n)
  return some (mkAppN sub.getAppFn next)

private def natConstant? (e : Expr) : Option Expr :=
  if e.isAppOfArity ``Op.constNat 1 then some e.getAppArgs[0]! else none

private def normalizeIndex (e : Expr) : Option Expr := Id.run do
  unless e.isAppOf ``Op.mul do return none
  let args := e.getAppArgs
  if args.size < 2 then return none
  let left := args[args.size-2]!
  let right := args[args.size-1]!
  let some b := natConstant? right | return none
  if let some a := natConstant? left then
    return some (mkApp (Lean.mkConst ``Op.constNat) (mkApp2 (Lean.mkConst ``Nat.mul) a b))
  unless left.isAppOf ``Op.mul do return none
  let inner := left.getAppArgs
  if inner.size < 2 then return none
  let some a := natConstant? inner[inner.size-1]! | return none
  let next := args.set! (args.size-2) inner[inner.size-2]!
    |>.set! (args.size-1) (mkApp (Lean.mkConst ``Op.constNat) (mkApp2 (Lean.mkConst ``Nat.mul) a b))
  return some (mkAppN e.getAppFn next)

elab_rules : term
  | `(source_nat_view $kernel:term) => do
    let source ← instantiateMVars (← elabTermEnsuringType kernel (Lean.mkConst ``ComputeKernel))
    let some source ← unfoldDefinition? source
      | throwErrorAt kernel "an index view requires a named source definition"
    return source.replace normalizeIndex
elab_rules : term
  | `(source_nat_range_view $kernel:term) => do
    let source ← instantiateMVars (← elabTermEnsuringType kernel (Lean.mkConst ``ComputeKernel))
    let some source ← unfoldDefinition? source
      | throwErrorAt kernel "an index view requires a named source definition"
    let result := ((source.replace normalizeBackward).replace normalizeBackward).replace normalizeIndex
    let resultType ← inferType result
    unless (← isDefEq resultType (Lean.mkConst ``ComputeKernel)) do
      throwErrorAt kernel "index view has an invalid source type"
    return result
end VeriTile.Meta
