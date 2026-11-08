import VeriTile.Triton.DSL.PureJitFunctions
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PureJitFunctions
open VeriTile.Triton
noncomputable section

def helper : PureJitFunction := triton_function (x) {
  return x * tl.sigmoid(x)
}

inductive Storage where | narrow | wide

def signature (left right : Storage) : CallSignature where
  pointers := [("X",match left with | .narrow => .fp32 | .wide => .fp64),
    ("B",match right with | .narrow => .fp32 | .wide => .fp64),
    ("Y",match right with | .narrow => .fp32 | .wide => .fp64)]
  constexpr := ["N"]
  readPorts := ["X","B"]
  writePorts := ["Y"]

def copied (left right : Storage) (X B Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call (signature left right) call_args [] pure_functions [silu := helper] in triton {
    off = tl.arange(0, N)
    x = tl.load(X + off).to(tl.float32)
    b = tl.load(B + off)
    y = silu(x) * b
    tl.store(Y + off, y)
  }

/-- The complete typed program fixes the helper's argument and return precision
before the caller's mixed-format multiplication and implicit destination cast. -/
theorem complete_program (left right : Storage) (X B Y : RegionName) (N : Nat) :
    copied left right X B Y N =
      (with_triton_call (signature left right) call_args [] in triton {
        off = tl.arange(0, N)
        x = tl.load(X + off).to(tl.float32)
        b = tl.load(B + off)
        __veritile_jit_0 = x
        __veritile_jit_1 = __veritile_jit_0 * tl.sigmoid(__veritile_jit_0)
        y = __veritile_jit_1 * b
        tl.store(Y + off, y)
      }) := rfl

/-- error: pure JIT helper cannot reference free names or other functions -/
#guard_msgs in
def freeName : PureJitFunction := triton_function (x) {
  return x + gain
}

/-- error: pure JIT helper requires one return expression -/
#guard_msgs in
def statements : PureJitFunction := triton_function (x) {
  y = x + 1
  return y
}

/-- error: pure JIT functions cannot be rebound, aliased or escape direct calls -/
#guard_msgs in
def rebound (X B Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call (signature .narrow .narrow) call_args [] pure_functions [silu := helper] in triton {
    silu = 0
  }

/-- error: pure JIT calls currently require a plain assignment -/
#guard_msgs in
def conditional (X B Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call (signature .narrow .narrow) call_args [] pure_functions [silu := helper] in triton {
    off = tl.arange(0, N)
    x = tl.load(X + off)
    if True:
        y = silu(x)
  }

/-- error: pure JIT call arguments must be register names -/
#guard_msgs in
def argumentExpression (X B Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call (signature .narrow .narrow) call_args [] pure_functions [silu := helper] in triton {
    off = tl.arange(0, N)
    y = silu(tl.load(X + off))
  }

/-- error: pure JIT helper cannot call other functions -/
#guard_msgs in
def nestedHelper : PureJitFunction := triton_function (x) {
  return silu(x)
}

/-- error: pure JIT calls cannot share a statement with conditional expressions or memory operations -/
#guard_msgs in
def conditionalExpression (X B Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call (signature .narrow .narrow) call_args [] pure_functions [silu := helper] in triton {
    off = tl.arange(0, N)
    x = tl.load(X + off)
    y = (silu(x) if True else x)
  }

/-- The callee remains FP32 while the caller's mixed FP32/FP64 product is FP64.
These checks inspect precision tags independently of the source comparison. -/
private def assignmentPrecision : ComputeStmt → Option ComputeDType
  | .assign _ _ _ (.compute (.alg precision _)) => some precision
  | _ => none
#guard ((copied .narrow .wide "X" "B" "Y" 2).surfaceBody[5]?).bind assignmentPrecision == some .fp32
#guard ((copied .narrow .wide "X" "B" "Y" 2).surfaceBody[6]?).bind assignmentPrecision == some .fp64

#axiomsClean complete_program
end
end VeriTile.Tests.PureJitFunctions
