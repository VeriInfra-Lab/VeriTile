import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.StringCallControl

namespace VeriTile.Tests.StringCallControl
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  constexpr := ["mode"]

/-- Both the initialization and the loop guard use the same immutable caller
string. Runtime conditions and their false branches are still executed. -/
def source (mode : String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    result = tl.full((), 0, tl.int64)
    if mode == "three":
        increment = tl.full((), 3, tl.int64)
    elif "one" == mode:
        increment = tl.full((), 1, tl.int64)
    else:
        increment = tl.full((), 7, tl.int64)
    for i in range(0, 2):
        if mode != "three":
            result = result + increment
        else:
            result = result + increment + tl.cast(i, tl.int64)
    if tl.program_id(0) == 0:
        result = result + tl.full((), 10, tl.int64)
  }

theorem actual_value {α : Type} [Inhabited α] (M : Algebra α) (mode : String) (s : State α) :
    ∃ t, exec M (source mode) s=some t ∧
      t.regs .int [] "result"=some (fun _ =>
        (if mode=="three" then 7 else if mode=="one" then 2 else 14) +
        if s.pids 0==0 then 10 else 0) := by
  by_cases ht : mode=="three" <;> by_cases ho : mode=="one"
  all_goals by_cases hp : s.pids 0=0
  all_goals simp [source,ht,ho,hp,exec,run,step,range,rangeInt,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat,
    numeric,bop,natLt,numericLt,numericEq]
  all_goals rfl

/-- error: string constexpr parameters must remain immutable -/
#guard_msgs in
def rebound (mode : String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    mode = "three"
  }
/-- error: a loop cannot shadow a string constexpr parameter -/
#guard_msgs in
def shadowed (mode : String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    for mode in range(0, 2):
        value = 0
  }
/-- error: string constexpr parameters support only equality or inequality with a literal string -/
#guard_msgs in
def escaping (mode : String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = mode
  }
/-- error: string constexpr parameters support only equality or inequality with a literal string -/
#guard_msgs in
def freeComparison (mode other : String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    if mode == other:
        value = 0
  }
/-- error: duplicate name in call signature -/
#guard_msgs in
def duplicate (mode : String) : ComputeKernel :=
  with_triton_call ({constexpr := ["mode","mode"]} : CallSignature) call_args [] in triton {
    value = 0
  }

#axiomsClean actual_value
end
end VeriTile.Tests.StringCallControl
