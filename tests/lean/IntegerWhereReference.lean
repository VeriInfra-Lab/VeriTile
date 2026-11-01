import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.IntegerWhereReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("L",.i32),("Out",.i32),("X",.fp32),("Y",.fp32)]
  readPorts := ["L","X"]
  writePorts := ["Out","Y"]

def integerReference (L Out X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    n = tl.load(L)
    bound = tl.cast(tl.where(n > 0, tl.cast(1, tl.int64), tl.cast(0, tl.int64)), tl.int64)
    for i in range(0, bound, 1):
        tl.store(Out, i)
    return
  }

def floatingReference (L Out X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = tl.load(X)
    y = tl.where(x > 0.0, 1.0, 0.0)
    tl.store(Y, y)
    return
  }
end VeriTile.Tests.IntegerWhereReference
