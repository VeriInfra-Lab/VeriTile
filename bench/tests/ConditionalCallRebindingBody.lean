import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ConditionalCallRebindingBody
open VeriTile.Triton FP.Structural
set_option maxRecDepth 20000
set_option maxHeartbeats 800000

def signature : CallSignature where
  scalars := [("x",.integer)]
  constexpr := ["FLAG"]

def source (FLAG : Bool) (value : Int) : ComputeKernel :=
  with_triton_call signature call_args [x := value] in triton {
    if FLAG:
        x = x + 1
    y = x + 2
  }

def expected (FLAG : Bool) (value : Int) : ComputeKernel := triton {
  if $((FLAG : Bool)) {
    x = $((value : Int)) + $(1)
    y = x + $(2)
  } else {
    y = $((value : Int)) + $(2)
  }
}

/-- The unassigned arm retains the caller's input; the other arm reads its
new local value. The whole continuation, not only the assignment, is checked. -/
theorem complete_conditional_binding (FLAG : Bool) (value : Int) :
    source FLAG value = expected FLAG value := rfl

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def loopMutation (FLAG : Bool) (value : Int) : ComputeKernel :=
  with_triton_call signature call_args [x := value] in triton {
    for i in range(2):
        x = x + 1
    y = x
  }

#axiomsClean complete_conditional_binding
end VeriTile.Tests.ConditionalCallRebindingBody
