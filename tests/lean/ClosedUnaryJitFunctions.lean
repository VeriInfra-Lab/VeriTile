import VeriTile.Triton.DSL.ClosedUnaryJitFunctions
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ClosedUnaryJitFunctions
open VeriTile.Triton
noncomputable section

def inner : ClosedUnaryJitFunction := triton_function (x) constexpr_args [] {
    """Scale the input."""
    return x * factor
}
def outer : ClosedUnaryJitFunction := triton_function (x) constexpr_args [] {
    return (x + inner(x))
}
def profile : CallSignature where
  pointers := [("input",.fp32),("output",.fp32)]
  sourceConstants := [("factor","0.5")]
  readPorts := ["input"]
  writePorts := ["output"]
def copied (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] jit_functions [inner := inner, outer := outer] in triton {
    value = outer(tl.load(input))
    tl.store(output, value)
  }
def captured (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    __veritile_closed_jit_0 = tl.load(input)
    __veritile_closed_jit_1 = __veritile_closed_jit_0
    __veritile_closed_jit_2 = __veritile_closed_jit_1 * factor
    __veritile_closed_jit_3 = __veritile_closed_jit_0 + __veritile_closed_jit_2
    value = __veritile_closed_jit_3
    tl.store(output, value)
  }
theorem nested_capture_exact (input output : RegionName) : copied input output=captured input output := by rfl
#axiomsClean nested_capture_exact

def cycle : ClosedUnaryJitFunction := triton_function (x) constexpr_args [] {
    return cycle(x)
}
/-- error: recursive JIT helper closures are unsupported -/
#guard_msgs in
example (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] jit_functions [cycle := cycle] in triton {
    value = cycle(tl.load(input))
    tl.store(output, value)
  }

def capture : ClosedUnaryJitFunction := triton_function (x) constexpr_args [] {
    return x + caller_value
}
/-- error: JIT helper cannot capture an unbound name: caller_value -/
#guard_msgs in
example (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] jit_functions [capture := capture] in triton {
    caller_value = tl.load(input)
    value = capture(caller_value)
    tl.store(output, value)
  }
def flags : CallSignature := {profile with constexpr := ["enabled"]}
def retyped (input output : RegionName) (enabled : Bool) : ComputeKernel :=
  with_triton_call flags call_args [] jit_functions [inner := inner, outer := outer] in triton {
    if enabled:
        input = tl.load(input)
        value = outer(input)
        tl.store(output, value)
  }
def flat (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] jit_functions [inner := inner, outer := outer] in triton {
    input = tl.load(input)
    value = outer(input)
    tl.store(output, value)
  }
def empty (input output : RegionName) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
  }
/-- Compile-time selection preserves the exact complete selected program,
including a source pointer parameter rebound to its loaded tensor. -/
theorem constexpr_retyping_exact (input output : RegionName) (enabled : Bool) :
    retyped input output enabled=(if enabled then flat input output else empty input output) := by rfl
#axiomsClean constexpr_retyping_exact

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.ClosedUnaryJitFunctions
