import VeriTile.Triton.DSL.PythonShapeExpressions
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PythonShapeExpressions
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["A","B"]
  writePorts := ["Y"]

def source (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    off = tl.program_id(0) << 5
    p = tl.make_block_ptr(Y, (A * B, B), (B, 1), (off, 0), (1, B), (1, 0))
    values = tl.full([1, B], 0, tl.float32)
    tl.store(p, values, boundary_check=(0,))
  }

/-- Shape products are metadata; the executable shift and dynamic block
pointer offset remain ordinary typed instructions in the complete program. -/
theorem complete_program (Y : RegionName) (N A B : Nat) :
    source Y N A B = (triton {
      tl.ports inputs=[], outputs=[Y]
      tl.region Y = tl.float32
      off = tl.program_id(0) << $((5 : Nat))
      p = tl.make_block_ptr(base=Y, shape=($((A * B : Nat)), $(B)),
        strides=($(B), 1), offsets=(off, 0), block_shape=(1, $(B)), order=(1, 0))
      values = tl.full([1, $(B)], 0, dtype=tl.float32)
      tl.store(p, values, boundary_check=([0] : List Nat))
    }) := rfl

/-- error: call dimensions used by shape expressions must remain immutable -/
#guard_msgs in
def rebound (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    A += 1
  }

/-- error: a loop cannot shadow a call dimension -/
#guard_msgs in
def shadowed (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    for B in range(0, N):
        x = 0
  }

/-- error: shape expression cannot read a runtime local or unknown metadata -/
#guard_msgs in
def runtime_shape (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    x = tl.program_id(0)
    p = tl.make_block_ptr(Y, (x * B, B), (B, 1), (0, 0), (1, B), (1, 0))
  }

/-- error: static shape assertion is not established by the caller -/
#guard_msgs in
def false_assertion (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    tl.static_assert(B == 32)
  }

/-- error: static assertion cannot depend on runtime scalar arguments -/
#guard_msgs in
def runtime_assertion (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    tl.static_assert(N == 32)
  }

/-- error: static shape assertions must occur at the top level of the copied body -/
#guard_msgs in
def nested_assertion (Y : RegionName) (N A B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [N := N] in triton {
    if N > 0:
        tl.static_assert(B == 32)
  }

#axiomsClean complete_program
end
end VeriTile.Tests.PythonShapeExpressions
