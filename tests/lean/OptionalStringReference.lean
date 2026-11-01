import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.OptionalStringReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["MODE"]
  readPorts := ["X"]
  writePorts := ["Y"]

def reference (X Y : RegionName) (MODE : Option String) : ComputeKernel :=
  let HAS : Bool := MODE.isSome
  let QK : Bool := MODE == some "qk"
  with_triton_call {signature with constexpr := []} call_args [] in triton {
    x = tl.load(X)
    if HAS:
        if QK:
            x = x + 1.0
        else:
            x = x + 2.0
    else:
        x = x + 3.0
    tl.store(Y, x)
  }
end VeriTile.Tests.OptionalStringReference
