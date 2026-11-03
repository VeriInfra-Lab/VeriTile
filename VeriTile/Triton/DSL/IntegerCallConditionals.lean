import VeriTile.Triton.DSL.IntegerCallSyntax
import VeriTile.Triton.DSL.PythonConditionGrouping

/-! Compose existing Python assignment-conditional grouping with checked
integer/arange invocation metadata. No source statement is removed or reordered. -/
namespace VeriTile.Triton.DSL.IntegerCallConditionals
open Lean Elab Term
@[term_elab IntegerCallArguments.withIntegers]
def elaborateInvocation : TermElab := fun stx expected => do
  IntegerCallSyntax.elaborateInvocation (← PythonConditionGrouping.prepareSource stx) expected
end VeriTile.Triton.DSL.IntegerCallConditionals
