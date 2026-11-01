import VeriTile.Triton.DSL.PythonShapeTuples
namespace VeriTile.Tests.NamedFloatDTypeCastReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Independent complete literal programs, compiled before the new adapter.
The source cdiv uses the standard-library signed expression, with no extra
integer cast. Every load mask, explicit float cast and output cast remains. -/
def reference (X Y : RegionName) (N : Int) (B : Nat) : FloatDType → ComputeKernel
  | .real =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.float64)
        tl.store(Y, value)
    }
  | .fp32 =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.float32)
        tl.store(Y, value)
    }
  | .fp16 =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.float16)
        tl.store(Y, value)
    }
  | .bf16 =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.bfloat16)
        tl.store(Y, value)
    }
  | .f8e4 =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.float8e4nv)
        tl.store(Y, value)
    }
  | .f8e5 =>
    with_triton_call signature call_args [N := N] in triton {
      for row in range(0, ((N + (B - 1)) // B)):
        value = tl.load(X, mask=row < N, other=0.0).to(tl.float8e5)
        tl.store(Y, value)
    }
end VeriTile.Tests.NamedFloatDTypeCastReference
