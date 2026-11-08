import VeriTile.Triton.DSL.StatementJitFunctions
import VeriTile.Triton.DSL.IntegralBooleanOperands
import VeriTile.Triton.DSL.LocalWeakIntegerConstants

/-! Explicit integer promotion of Boolean operands in a complete pasted
statement-helper invocation. The ordinary helper frontend retains lexical
scope, tuple returns, scalar bindings and all floating operations. -/
namespace VeriTile.Triton.DSL.IntegralJitFunctions
open Lean Elab Term
syntax (name := integralJitCall) "with_integral_operands " "in " term : term
@[term_elab integralJitCall]
def elaborateCall : TermElab := fun stx expected=>do
  let call:=stx[2]
  unless call.isOfKind ``StatementJitFunctions.withFunctions do
    throwError "a complete statement-helper invocation is required"
  StatementJitFunctions.elaborateInvocation
    (← IntegralBooleanOperands.prepare (← LocalWeakIntegerConstants.prepare call)) expected
end VeriTile.Triton.DSL.IntegralJitFunctions
