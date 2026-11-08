import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Tests.SourceModuleBindings
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  scalars := []
  constexpr := ["MODE"]
  sourceConstants := [("_ZERO","0")]
  unaryImports := [("tanh","triton.language.extra.libdevice.tanh")]

def kernel (x y : RegionName) (MODE : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(x)
    if MODE == _ZERO:
        value = tanh(value)
    tl.store(y, value)
  }

/-- Compare the complete lowered source against an explicitly named libdevice
call. In particular a `tl.tanh` call cannot pass as the imported intrinsic. -/
def explicitKernel (x y : RegionName) (MODE : Nat) : ComputeKernel :=
  triton {
    tl.call_signature
    tl.region x = tl.float32
    tl.region y = tl.float32
    value = tl.load(x)
    if MODE == 0 {
        value = tl.extra.cuda.libdevice.tanh(value)
    }
    tl.store(y, tl.implicit_store_cast(value, tl.float32))
  }

example (x y : RegionName) (MODE : Nat) : kernel x y MODE = explicitKernel x y MODE := by rfl

/-- error: source constant requires an integer literal -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  let bad : CallSignature := { signature with sourceConstants := [("_ZERO","1-1")] }
  with_triton_signature bad in triton { tl.store(y, 0.0) }

/-- error: source intrinsic aliases may only occur in direct calls and cannot be rebound -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    tanh = tl.load(x)
    tl.store(y, tanh)
  }

end VeriTile.Tests.SourceModuleBindings
