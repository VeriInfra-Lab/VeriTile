import VeriTile.Triton.Float.Erasure
import Lean

/-! A proof-only mathematical view computed from a named source. This embeds
its existing Algorithm projection; it does not delete/reorder source statements
or replace a numerical operation. Canonical kernels must not use this helper.
The result is definitionally equal to the ordinary projection/embedding. -/
namespace VeriTile.Meta
open Lean Elab Term Meta
open VeriTile.Triton
syntax "source_algorithm_view " term : term
elab_rules : term
  | `(source_algorithm_view $source:term) => do
    let source ← instantiateMVars (← elabTermEnsuringType source (mkConst ``ComputeKernel))
    unless source.getAppFn.isConst do throwError "a mathematical view requires a named kernel source"
    let .defnInfo _ ← getConstInfo source.getAppFn.constName!
      | throwError "a mathematical view requires a named kernel definition"
    let checked ← whnf (← mkAppM ``ComputeKernel.toAlgorithm? #[source])
    unless checked.isAppOf ``Except.ok do throwError "source has no complete Algorithm projection"
    let algorithm := checked.getAppArgs.back!
    let inputs ← mkAppM ``Kernel.inputs #[algorithm]
    let outputs ← mkAppM ``Kernel.outputs #[algorithm]
    let body ← mkAppM ``Kernel.body #[algorithm]
    let embedded ← mkAppM ``ComputeKernel.fromKernelBody #[inputs,outputs,body]
    let normalized ← reduce embedded (explicitOnly:=false)
    let normalized ← Lean.Meta.transform normalized (post:=fun expression => do
      if expression.isAppOfArity ``Region.mk 2 then
        let field := expression.getAppArgs[1]!
        let original? := match field with
          | .proj ``Region 0 value => some value
          | _ => if field.isAppOfArity ``Region.name 2 then some field.getAppArgs[1]! else none
        if let some original := original? then
          if ← isDefEq (← inferType original) (← inferType expression) then
            return .done original
      return .done expression)
    unless ← isDefEq normalized embedded do
      throwError "mathematical view is not definitionally equal to the source projection"
    return normalized
end VeriTile.Meta
