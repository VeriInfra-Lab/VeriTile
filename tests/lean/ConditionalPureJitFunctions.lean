import VeriTile.Triton.DSL.ConditionalPureJitFunctions
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ConditionalPureJitFunctions
open VeriTile.Triton
noncomputable section

def leaky : PureJitFunction := triton_function (x) {
    """The source docstring is kept with the helper."""
    return tl.where(x >= 0, x, 0.01 * x)
  }

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  scalars := [("M",.integer),("N",.integer)]
  constexpr := ["BLOCK","ACTIVATION"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (M N BLOCK : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N]
    pure_functions [leaky_relu := leaky] in triton {
    M, N = tl.cast(M, tl.int64), tl.cast(N, tl.int64)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + cols, mask=cols < N, other=0.0)
    if ACTIVATION == "leaky_relu":
        values = leaky_relu(values)
    tl.store(Y + cols, values, mask=cols < N)
  }

/-- Independent complete quotation: all self-casts remain statements and
helper argument/result assignments stay inside the original activation branch.
The FP16 destination conversion is checked by the same explicit caller ABI. -/
theorem complete_program (X Y : RegionName) (M N BLOCK : Nat) (ACTIVATION : String) :
    source X Y M N BLOCK ACTIVATION =
      (with_triton_call signature call_args [M := M,N := N] in triton {
        M = tl.cast(M, tl.int64)
        N = tl.cast(N, tl.int64)
        cols = tl.arange(0, BLOCK)
        values = tl.load(X + cols, mask=cols < N, other=0.0)
        if ACTIVATION == "leaky_relu":
            __veritile_jit_control_0 = values
            __veritile_jit_control_1 = tl.where(__veritile_jit_control_0 >= 0,
                __veritile_jit_control_0, 0.01 * __veritile_jit_control_0)
            values = __veritile_jit_control_1
        tl.store(Y + cols, values, mask=cols < N)
      }) := by rfl

/-- error: pure JIT calls require a complete plain assignment at the call site -/
#guard_msgs in
def nested_call (X Y : RegionName) (M N BLOCK : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N]
    pure_functions [leaky_relu := leaky] in triton {
    values = tl.load(X)
    values = 1.0 + leaky_relu(values)
  }

/-- error: pure JIT calls require a complete plain assignment at the call site -/
#guard_msgs in
def loop_call (X Y : RegionName) (M N BLOCK : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N]
    pure_functions [leaky_relu := leaky] in triton {
    values = tl.load(X)
    for i in range(0, BLOCK):
        values = leaky_relu(values)
  }

/-- A manually constructed source descriptor is still validated. -/
def invalidHelper : PureJitFunction := ⟨"x","return unknown_name"⟩

/-- error: pure JIT helper cannot reference free names or other functions -/
#guard_msgs in
def free_name (X Y : RegionName) (M N BLOCK : Nat) (ACTIVATION : String) : ComputeKernel :=
  with_triton_call signature call_args [M := M,N := N]
    pure_functions [leaky_relu := invalidHelper] in triton {
    values = tl.load(X)
    values = leaky_relu(values)
  }

#axiomsClean complete_program
end
end VeriTile.Tests.ConditionalPureJitFunctions
