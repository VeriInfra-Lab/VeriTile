import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.IntegerMinMaxReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("I",.i32),("F",.fp32),("Out",.i32)]
  scalars := [("window",.integer)]
  constexpr := ["B"]

def integer (I F Out : RegionName) (window B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [window := window] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(I + offsets)
    y = tl.maximum(x - window, tl.cast(0, tl.int64))
    tl.store(Out + offsets, y)
  }

def floating (I F Out : RegionName) (window B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [window := window] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(I + offsets)
    x = tl.load(F + offsets)
    y = tl.maximum(x, 0)
    tl.store(Out + offsets, y.to(tl.int32))
  }
end VeriTile.Tests.IntegerMinMaxReference
