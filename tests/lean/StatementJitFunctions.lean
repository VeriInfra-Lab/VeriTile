import VeriTile.Triton.DSL.StatementJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StatementJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def pair : StatementJitFunction :=
  triton_function (x, y) constexpr_args [] {
    total = x + y
    return total, x
  }
def signature : CallSignature where
  scalars := [("a",.fp32),("b",.fp32)]

/-- Keyword order differs from parameter order; caller destinations alias the
arguments and the chosen temporary spelling is already occupied. -/
def source : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    __veritile_statement_jit_0 = tl.full((), 17, tl.int64)
    a, b = pair(y=b, x=a)
    c, d = pair(a, b)
  }

theorem simultaneous_and_repeated {α : Type} [Inhabited α] (M : Algebra α)
    (a b : Values α .real []) (s : State α)
    (ha : s.regs .real [] "a"=some a) (hb : s.regs .real [] "b"=some b) :
    let add := M.binary (some .fp32) .real .add
    ∃ t, exec M source s=some t ∧
      t.regs .real [] "a"=some (fun i=>add (a i) (b i)) ∧
      t.regs .real [] "b"=some a ∧
      t.regs .real [] "c"=some (fun i=>add (add (a i) (b i)) (a i)) ∧
      t.regs .int [] "__veritile_statement_jit_0"=some (fun _=>17) := by
  simp [source,ha,hb,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat]
  exact ⟨rfl,rfl⟩

def lanes : StatementJitFunction :=
  triton_function (x, N) constexpr_args [N] {
    HALF: tl.constexpr = N // 2
    indices = tl.arange(0, HALF).to(tl.int64)
    return indices + x
  }
def integerSignature : CallSignature where
  scalars := [("x",.integer)]
  constexpr := ["N"]
def laneSource (x : Int) (N : Nat) : ComputeKernel :=
  with_triton_call integerSignature call_args [x := x]
    jit_functions [lanes := lanes] in triton {
    result = lanes(x, N)
  }

theorem constexpr_and_signed_cast {α : Type} [Inhabited α] (M : Algebra α)
    (x : Int) (N : Nat) (s : State α) :
    ∃ t, exec M (laneSource x N) s=some t ∧
      t.regs .int [N/2] "result"=some (fun i=>(i.1.val : Int)+x) := by
  simp [laneSource,exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,State.setReg,ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

/-- error: JIT helper references an unbound name: secret -/
#guard_msgs in
def freeName : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    y = x + secret
    return y
  }
/-- error: JIT helper needs one final scalar or tuple return -/
#guard_msgs in
def noReturn : StatementJitFunction :=
  triton_function (x) constexpr_args [] { y = x + 1 }
/-- error: duplicate JIT argument: x -/
#guard_msgs in
def duplicateArgument : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    c, d = pair(a, x=b)
  }
/-- error: JIT return arity does not match assignment -/
#guard_msgs in
def wrongArity : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    c = pair(a, b)
  }
/-- error: a helper constexpr cannot receive a runtime register -/
#guard_msgs in
def runtimeSize (x : Int) (N : Nat) : ComputeKernel :=
  with_triton_call integerSignature call_args [x := x]
    jit_functions [lanes := lanes] in triton { result = lanes(x, x) }
/-- error: JIT helpers may occur only in direct, plain assignments -/
#guard_msgs in
def aliasFunction : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    escaped = pair
  }
/-- error: JIT helper references an unbound name: secret -/
#guard_msgs in
def forgedMetadata : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions
    [pair := (⟨["x","y"],[],"return secret, x"⟩ : StatementJitFunction)] in triton {
    c, d = pair(a, b)
  }

/-- error: a JIT constexpr cannot be rebound -/
#guard_msgs in
def reboundLocalConstant : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    DIM: tl.constexpr = 128
    DIM = 64
    return x
  }
/-- error: caller constexpr arguments must remain immutable -/
#guard_msgs in
def reboundCallerConstant (x : Int) (N : Nat) : ComputeKernel :=
  with_triton_call integerSignature call_args [x := x]
    jit_functions [lanes := lanes] in triton {
    N = 4
    result = lanes(x, N)
  }
/-- error: unknown JIT keyword argument: other -/
#guard_msgs in
def unknownKeyword : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    c, d = pair(x=a, other=b)
  }
/-- error: JIT helpers may occur only in direct, plain assignments -/
#guard_msgs in
def conditionalCall : ComputeKernel :=
  with_triton_call signature call_args [] jit_functions [pair := pair] in triton {
    if a > b:
        c, d = pair(a, b)
  }

#axiomsClean simultaneous_and_repeated
#axiomsClean constexpr_and_signed_cast
end
end VeriTile.Tests.StatementJitFunctions
