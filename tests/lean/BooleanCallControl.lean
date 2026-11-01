import VeriTile.Triton.DSL.BooleanCallControl
import tests.lean.BooleanCallControlReference
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.BooleanCallControl
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["B","ITER","READ_MASK","WRITE_MASK"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    lanes = tl.arange(0, B)
    if READ_MASK:
        active = lanes < N
        x = tl.load(X + lanes, active, other=0.0)
    else:
        x = tl.load(X + lanes)
    if WRITE_MASK:
        mask_c = lanes < N
    else:
        mask_c = None
    for _ in range(ITER):
        tl.store(Y + lanes, x, mask_c)
  }

theorem complete_program (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) :
    copied X Y N B ITER READ_MASK WRITE_MASK =
      BooleanCallControlReference.program X Y N B ITER READ_MASK WRITE_MASK := by
  cases READ_MASK <;> cases WRITE_MASK <;> rfl

/-- error: Boolean constexpr arguments must remain immutable -/
#guard_msgs in
def mutated (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    READ_MASK = False
  }

/-- error: a loop cannot shadow a Boolean constexpr argument -/
#guard_msgs in
def shadowed (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    for WRITE_MASK in range(ITER):
        tl.store(Y, 0.0)
  }

/-- error: None mask aliases must be fresh compile-time bindings -/
#guard_msgs in
def runtimeNone (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    if tl.program_id(0) == 0:
        mask = None
    tl.store(Y, 0.0, mask)
  }

/-- error: None mask aliases cannot be rebound -/
#guard_msgs in
def reboundNone (X Y : RegionName) (N B ITER : Nat) (READ_MASK WRITE_MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    mask = None
    mask = tl.program_id(0) < N
    tl.store(Y, 0.0, mask)
  }

#axiomsClean complete_program
#auditModuleAxioms
end VeriTile.Tests.BooleanCallControl
