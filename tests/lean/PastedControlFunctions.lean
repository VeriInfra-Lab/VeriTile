import VeriTile.Triton.DSL.PastedControlFunctions
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PastedControlFunctions
open VeriTile.Triton DSL.PastedControlFunctions

def signature : CallSignature where
  scalars := [("x",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","ACC_TYPE"]

def sample : Invocation :=
  with_control_function "caller" signature constexpr_bools []
    dtype_constants [ACC_TYPE := .fp32] jit_calls [callee := "map"] in triton {
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    row, col = callee(x, BLOCK_M, BLOCK_N)
  }

theorem entrypoint_preserved : sample.entrypoint="caller" := rfl
theorem dtype_binding : sample.dtypeConstants=[("ACC_TYPE",.fp32)] := rfl
theorem helper_binding : sample.helperBindings=[("callee","map")] := rfl
theorem complete_body : sample.body.length=2 := rfl

theorem call_retained :
    (match sample.body[1]? with
    | some (VeriTile.Triton.ControlSource.Statement.bindCall targets name args) =>
      targets=["row","col"] ∧ name="callee" ∧ args.length=3
    | _ => False) := by exact ⟨rfl,rfl,rfl⟩

#axiomsClean entrypoint_preserved
#axiomsClean dtype_binding
#axiomsClean helper_binding
#axiomsClean complete_body
#axiomsClean call_retained
end VeriTile.Tests.PastedControlFunctions
