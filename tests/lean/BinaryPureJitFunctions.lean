import tests.lean.BinaryPureJitReference
import VeriTile.Triton.DSL.BinaryPureJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.BinaryPureJitFunctions
open VeriTile.Triton FP.Structural BinaryPureJitReference
noncomputable section

def previous : BinaryPureJitFunction := triton_function (a, b) {
    return tl.cdiv(a, b) * b - b
}

def source (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] binary_functions [prev := previous] in triton {
    previous = prev(N, B)
    tl.store(Out, previous)
  }

theorem no_added_conversion : source=reference := rfl

/-- In particular the zero-length helper result is negative; no natural
subtraction or erased callback can turn it into zero. -/
theorem zero_previous {A : Type} [Inhabited A] (M : Algebra A) (s : State A) :
    (exec M (source "out" 0 4) s).map (fun t=>(t.mem "out" 0).read .int)=some (-4 : Int) := by
  simp [source,previous,ComputeKernel.surfaceBody,exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,numericLt,natLt,
    integralFloorDiv,Region.cast,State.setReg,store]

def repeated (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] binary_functions [prev := previous] in triton {
    __veritile_binary_jit_0 = N
    first = prev(N, B)
    second = prev(N, B)
    tl.store(Out, first + second + __veritile_binary_jit_0)
  }

theorem fresh_captures : repeated=repeatedReference := rfl

/-- error: binary JIT helper supports only its two parameters, integer literals and integer arithmetic -/
#guard_msgs (error) in
 def badFree : BinaryPureJitFunction := triton_function (a, b) { return a + other }

/-- error: binary JIT helper supports only its two parameters, integer literals and integer arithmetic -/
#guard_msgs (error) in
 def badMemory : BinaryPureJitFunction := triton_function (a, b) { return tl.load(a) + b }

/-- error: binary JIT argument typing requires immutable integer call and constexpr parameters -/
#guard_msgs (error) in
 def badMutation (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] binary_functions [prev := previous] in triton {
    B = 2
    value = prev(N, B)
    tl.store(Out, value)
  }

/-- error: a loop cannot shadow a binary JIT integer parameter -/
#guard_msgs (error) in
 def badShadow (Out : RegionName) (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] binary_functions [prev := previous] in triton {
    for N in range(0, 2):
      value = 1
    value = prev(N, B)
    tl.store(Out, value)
  }

/-- error: binary JIT result requires a signed caller argument in its integer expression -/
#guard_msgs (error) in
 def badNatural (Out : RegionName) (N : Nat) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] binary_functions [prev := previous] in triton {
    value = prev(N, B)
    tl.store(Out, value)
  }

#axiomsClean fresh_captures
#axiomsClean no_added_conversion
#axiomsClean zero_previous
end
end VeriTile.Tests.BinaryPureJitFunctions
