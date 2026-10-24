import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("count",.i32),("stride",.i64)]
  constexpr := ["B"]

def pasted (X Y : RegionName) (B count : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [count := count, stride := stride] in
  triton {
    """A Python docstring can contain } $ tl.region and
def fake_definition:
end fake_namespace
    without terminating the quotation."""
    # } $ tl.param and -- are ordinary Python comment text.
    offsets = tl.arange(0, B).to(tl.int64)
    active = offsets < count
    x = tl.load(
        X + offsets * stride,
        mask=active, other=0,
    ).to(tl.float32)
    if count > 0:
        y = x--x
        tl.store(Y + offsets, y, mask=active)
    else:
        tl.store(Y + offsets, x, mask=active)
  }

def expanded (X Y : RegionName) (B count : Nat) (stride : Int) : ComputeKernel :=
  triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    offsets = tl.arange(0, B).to(tl.int64)
    active = offsets < $((count : Nat))
    x = tl.load(X + offsets * $((stride : Int)), mask=active, other=0).to(tl.float32)
    if $((count : Nat)) > 0 {
      y = x - -x
      tl.store(Y + offsets, y, mask=active)
    } else {
      tl.store(Y + offsets, x, mask=active)
    }
  }

theorem original_comments_indentation_and_integer_calls
    (X Y : RegionName) (B count : Nat) (stride : Int) :
    pasted X Y B count stride = expanded X Y B count stride := rfl

def integerSignature : CallSignature where
  scalars := [("n",.i64)]

def scalarCast (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    value = n.to(tl.int64)
  }

/-- Signed values supplied by a call reach the signed register unchanged. -/
theorem signed_call_executes {A : Type} [Inhabited A] (M : FP.Structural.Algebra A)
    (s : FP.Structural.State A) (n : Int) :
    (FP.Structural.exec M (scalarCast n) s).bind
      (fun t => t.regs .int [] "value") = some (fun _ => n) := by
  simp [scalarCast, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalOp_unfold, FP.Structural.State.setReg]

theorem scalar_cast_is_retained (n : Int) : scalarCast n = triton {
    value = ($((n : Int))).to(tl.int64)
  } := rfl

/-- error: every integer runtime scalar needs an explicit call binding -/
#guard_msgs in
def missingBinding (n : Nat) : ComputeKernel :=
  with_triton_call signature call_args [count := n] in
  triton { value = count }

/-- error: duplicate call argument binding -/
#guard_msgs in
def duplicateBinding (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n, n := n] in
  triton { value = n }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def assignedArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n += 1
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def shadowedArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    for n in range(4):
        value = n
  }

/-- error: per-call specialization does not yet support reassigned scalar arguments -/
#guard_msgs in
def tupleArgument (n : Int) : ComputeKernel :=
  with_triton_call integerSignature call_args [n := n] in
  triton {
    n, other = 1, 2
  }

/-- error: observed memory access is missing from the declared call ports -/
#guard_msgs in
def incorrectMemoryRole (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)], readPorts := ["Y"], writePorts := ["Y"]} : CallSignature) in
  triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

#axiomsClean original_comments_indentation_and_integer_calls
#axiomsClean scalar_cast_is_retained
#axiomsClean signed_call_executes
end VeriTile.Tests.PythonBody
