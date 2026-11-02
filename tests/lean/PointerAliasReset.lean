import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerAliasReset
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  constexpr := ["B"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    offsets = tl.arange(0, B)
    p = X + offsets
    for j in range(0, 2):
        p = X + offsets
        x = tl.load(p)
        tl.store(Y + offsets, x)
  }

/-- A same-base reset preserves the pointee type and the independent FP16
store conversion; the loop may not discard or change either operation. -/
theorem complete_program (X Y : RegionName) (B : Nat) : source X Y B =
    (triton {
      tl.region X = tl.float32, Y = tl.float16
      offsets = tl.arange(0, $(B))
      p = X + offsets
      for j in range($(0), $(2), $(1)) {
        p = X + offsets
        x = tl.load(p)
        tl.store(Y + offsets, x.to(tl.float16))
      }
    }) := rfl

/-- error: pointer alias is rebound under runtime control flow -/
#guard_msgs in
example (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    offsets = tl.arange(0, B)
    p = X + offsets
    for j in range(0, 2):
        p = Y + offsets
        tl.store(p, 0.0)
  }

/-- error: pointer alias is rebound under runtime control flow -/
#guard_msgs in
example (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    offsets = tl.arange(0, B)
    p = X + offsets
    for j in range(0, 2):
        p = 0
        tl.store(Y + offsets, 0.0)
  }

#axiomsClean complete_program
end
end VeriTile.Tests.PointerAliasReset
