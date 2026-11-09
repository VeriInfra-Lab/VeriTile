import VeriTile.Triton.DSL.PythonInlineSuites
namespace VeriTile.Tests.PythonFullConstructorReference
open VeriTile.Triton

def constructors (B : Nat) : ComputeKernel := triton {
  scalar = tl.full([], 1, dtype=tl.int64)
  ints = tl.full([B], 3, dtype=tl.int64)
  floats = tl.full([B], 0.0, dtype=tl.float32)
}

def signedCasts (X : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[]
  tl.region X = tl.float32
  tl.param base : tl.int64
  pid = tl.program_id(0).to(tl.int64) + base
  offset = tl.cast(pid, tl.int64) * B + tl.arange(0,B).to(tl.int64)
  value = tl.load(X + offset, dtype=tl.float32)
}
end VeriTile.Tests.PythonFullConstructorReference
