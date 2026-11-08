import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ReassignedIntegerPointers
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("A",.i8),("Y",.i8)]
  scalars := [("stride",.integer)]
  constexpr := ["B"]
  readPorts := ["A"]
  writePorts := ["Y"]

def source (A Y : RegionName) (stride B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [stride := stride] in triton {
    stride = tl.cast(stride, tl.int64)
    i = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, B)
    A = A + (i * stride + lanes)
    a = tl.load(A)
    tl.store(Y + lanes, a)
  }

/-- A pointer remains a pointer even when its parameter name is assigned and
its full displacement mixes signed and natural source operands. -/
theorem complete_program (A Y : RegionName) (stride B : Nat) :
    source A Y stride B = (triton {
      tl.ports inputs=[A], outputs=[Y]
      tl.region A = tl.int8, Y = tl.int8
      stride = tl.cast($(stride), tl.int64)
      i = tl.program_id(0).to(tl.int64)
      lanes = tl.arange(0, $(B))
      A = A + (i * tl.cast($(stride), tl.int64) + lanes)
      a = tl.load(A)
      tl.store(Y + lanes, a)
    }) := rfl

#axiomsClean complete_program
end
end VeriTile.Tests.ReassignedIntegerPointers
