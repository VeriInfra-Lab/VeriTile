import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.WelfordSource

namespace VeriTile.Tests.OrderedSumReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["XBLOCK","B"]
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Independent typed program: the FP32 tree applies after the complete
streaming accumulation, and the result retains both singleton axes. -/
def reference (X Y : RegionName) (N B : Nat) (tree : DSL.WelfordSource.Tree B) : ComputeKernel :=
  let XBLOCK : Nat := 1
  with_triton_call signature call_args [N := N] in triton {
    acc = tl.full([1, B], 0, tl.float32)
    for base in range(0, N, B):
      lane = base + tl.arange(0, B)[None, :]
      value = tl.load(X + lane, mask=lane < N, other=0.0).to(tl.float32)
      acc = acc + value
    total = tl.reshape(tl.sum(acc, axis=1, tree=$(tree), keep_dims=true), [1])[:, None]
    tl.store(Y + tl.arange(0, 1)[:, None], total, None)
  }
end VeriTile.Tests.OrderedSumReference
