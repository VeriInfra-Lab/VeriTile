import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Tests.ChainedSourceCastsReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["X"]

/-- Compiled using the existing frontend, before importing the chain adapter. -/
def expected (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    off = tl.arange(0, B)
    x = tl.load(X + off)
    y = ((x).to(tl.float16)).to(tl.float32)
    tl.store(X + off, y)
  }
end VeriTile.Tests.ChainedSourceCastsReference
