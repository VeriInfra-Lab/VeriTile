import VeriTile.Triton.DSL
import Lean

/-! Proof-only scalar specialization of an existing source quotation.

The view expands a named source definition and replaces reads of one scalar
register with a mathematical literal. It preserves statements, control flow,
casts, storage and every other expression. Canonical Kernels.lean files keep
the original source; they must not use this helper to construct authority.

This operation does not assert execution equivalence. A proof using the view
must check its connection to the actual runtime source and scalar ABI, including
whether the source writes or shadows the scalar. Such a connection is checked
for the recurrent-delta kernels by RuntimeBridge.literal_pair.
-/
namespace VeriTile.Meta
open Lean Lean.Meta Lean.Elab Lean.Elab.Term
open VeriTile.Triton

syntax "source_scalar_view " term " at " str " := " term : term

elab_rules : term
  | `(source_scalar_view $kernel:term at $name:str := $value:term) => do
    let source ← instantiateMVars (← elabTermEnsuringType kernel (mkConst ``ComputeKernel))
    let value ← instantiateMVars (← elabTermEnsuringType value (mkConst ``Real))
    let some source ← unfoldDefinition? source
      | throwErrorAt kernel "a scalar view requires a named source definition"
    let shape ← mkListLit (mkConst ``Nat) []
    let reference ← mkAppM ``Op.ref #[mkConst ``TileDType.real, shape, toExpr name.getString]
    unless (source.find? (· == reference)).isSome do
      throwErrorAt name "no FP32 scalar read named {name.getString} in this source definition"
    let literal ← mkAppM ``Op.const #[value]
    return source.replace fun node => if node == reference then some literal else none

end VeriTile.Meta
