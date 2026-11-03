import VeriTile.Triton.DSL.GuardedJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.GuardedJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def sameGuard : StatementJitFunction :=
  triton_function (x, FLAG) constexpr_args [FLAG] {
    acc = x
    if FLAG:
        value = x + 1
    if FLAG:
        acc = value
    return acc
  }

/-- error: JIT helper references an unbound name: value -/
#guard_msgs in
def unguardedRead : StatementJitFunction :=
  triton_function (x, FLAG) constexpr_args [FLAG] {
    if FLAG:
        value = x
    return value
  }

/-- error: JIT helper references an unbound name: value -/
#guard_msgs in
def changedRuntimeGuard : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    acc = x
    if x > 0:
        value = 1
    x = 1
    if x > 0:
        acc = value
    return acc
  }

/-- error: JIT helper references an unbound name: value -/
#guard_msgs in
def zeroTripEscape : StatementJitFunction :=
  triton_function (x, N) constexpr_args [N] {
    for i in range(0, N, 1):
        value = x
    return value
  }

def exchange : StatementJitFunction :=
  triton_function (a, b, FLAG) constexpr_args [FLAG] {
    if FLAG:
        a, b = b, a
    return a, b
  }

def integerSignature : CallSignature where
  scalars := [("x",.integer),("y",.integer)]
  constexpr := ["FLAG"]

def exchangeCaller (x y : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call integerSignature call_args [x := x,y := y]
    jit_functions [exchange := exchange] in triton {
    a, b = exchange(x, y, FLAG)
  }

theorem parallel_exchange {α : Type} [Inhabited α] (M : Algebra α)
    (x y : Nat) (flag : Bool) (s : State α) :
    ∃ t, exec M (exchangeCaller x y flag) s=some t ∧
      t.regs .nat [] "a"=some (fun _=>if flag then y else x) ∧
      t.regs .nat [] "b"=some (fun _=>if flag then x else y) := by
  cases flag <;> simp [exchangeCaller,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind]

def identityBody : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    value = x
    return value
  }

def floatSignature : CallSignature where
  scalars := [("x",.fp32)]

def floatCaller (x : ℝ) : ComputeKernel :=
  with_triton_call floatSignature call_args [x := x]
    jit_functions [identity := identityBody] in triton {
    output = identity(x)
  }

/-- A standalone argument copy reads the materialized FP32 ABI value.
No same-type cast identity or other numerical premise is assumed. -/
theorem fp32_alias {α : Type} [Inhabited α] (M : Algebra α) (x : ℝ) (s : State α) :
    ∃ t, exec M (floatCaller x) s=some t ∧
      t.regs .real [] "output"=some (fun _=>M.literal (some .fp32) .real x) := by
  simp [floatCaller,ComputeKernel.withScalarArguments,ScalarArgument.fp32,ScalarArgument.statement,
    exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType]

def integerZero : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    if x < 0:
        x = tl.full((), 0, tl.int64)
    return x
  }

def signedSignature : CallSignature where
  scalars := [("x",.i64)]

def zeroCaller (x : Int) : ComputeKernel :=
  with_triton_call signedSignature call_args [x := x]
    jit_functions [clamp := integerZero] in triton {
    output = clamp(x)
  }

/-- The original positional constructor produces an integer zero. Its
meaning is independent of every floating operation and literal callback. -/
theorem integer_zero {α : Type} [Inhabited α] (M : Algebra α) (x : Int) (s : State α) :
    ∃ t, exec M (zeroCaller x) s=some t ∧
      t.regs .int [] "output"=some (fun _=>if x<0 then 0 else x) := by
  by_cases h : x<0 <;>
    simp [zeroCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,numericLt,bop,h]

def castSignature : CallSignature where
  pointers := [("X",.fp16)]

def registerCastCaller (X : RegionName) : ComputeKernel :=
  with_triton_call castSignature call_args [] jit_functions [] in triton {
    v = tl.load(X)
    p = tl.full((), 0.0, tl.float32)
    output = p.to(v.dtype)
  }

def registerCastExplicit (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float16
  v = tl.load(X)
  p = tl.full([], 0.0, dtype=tl.float32)
  output = (p).to(tl.float16)
}

/-- Resolving the ambiguous source spelling retains the FP32-to-FP16 cast,
including its precision context and both source statements before it. -/
theorem register_dtype_conversion (X : RegionName) :
    registerCastCaller X=registerCastExplicit X := by rfl

#axiomsClean register_dtype_conversion
#axiomsClean integer_zero
#axiomsClean parallel_exchange
#axiomsClean fp32_alias
end
end VeriTile.Tests.GuardedJitFunctions
