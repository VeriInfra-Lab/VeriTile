import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Tests.RawBodyControlNames
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  scalars := []
  constexpr := ["N","B"]

/-- Both `_` and `do` are ordinary Python identifiers. The final return is
copied verbatim and ends the same function as falling through its last store. -/
def copied (x y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    do = tl.load(x)
    for _ in range(0, tl.cdiv(N, B)):
        do = do + 1.0
    tl.store(y, do)
    return
  }

/-- The explicit DSL spelling exposes the same register names and loop bound. -/
def explicitKernel (x y : RegionName) (N B : Nat) : ComputeKernel :=
  triton {
    tl.call_signature
    tl.region x = tl.float32
    tl.region y = tl.float32
    «do» = tl.load(x)
    for «_» in range($(0), $(((N+B-1)/B : Nat)), $(1)) {
      «do» = «do» + 1.0
    }
    tl.store(y, tl.implicit_store_cast(«do», tl.float32))
  }

example (x y : RegionName) (N B : Nat) : copied x y N B = explicitKernel x y N B := by rfl

/- A return before another statement must not be silently treated as a
function-terminal return. -/
/-- error: `return` is only supported as the sole body statement of `if cond { return }` -/
#guard_msgs in
example (x y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    return
    tl.store(y, tl.load(x))
  }

end VeriTile.Tests.RawBodyControlNames
