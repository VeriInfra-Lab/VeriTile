import tests.lean.OptionalStringReference
import VeriTile.Triton.DSL.OptionalStringConstants
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.OptionalStringConstants
open VeriTile.Triton OptionalStringReference

def original (X Y : RegionName) (MODE : Option String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = tl.load(X)
    if MODE is not None:
        if MODE == 'qk':
            x = x + 1.0
        else:
            x = x + 2.0
    else:
        x = x + 3.0
    tl.store(Y, x)
  }

theorem complete_program : original=reference := rfl

/-- error: optional string constexpr parameters must remain immutable -/
#guard_msgs in
example (X Y : RegionName) (MODE : Option String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    MODE = 0
    return
  }

/-- error: a loop cannot shadow an optional string constexpr -/
#guard_msgs in
example (X Y : RegionName) (MODE : Option String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    for MODE in range(1):
        tl.store(Y, 0.0)
  }

/-- error: optional strings support only None guards and equality with a literal string -/
#guard_msgs in
example (X Y : RegionName) (MODE : Option String) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = MODE
    return
  }

#axiomsClean complete_program
end VeriTile.Tests.OptionalStringConstants
