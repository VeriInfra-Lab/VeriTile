import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.AliasedNestedLoadBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp32)]
  scalars := [("scale",.fp32)]
  constexpr := ["N"]

def source (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    __veritile_source_load_0 = tl.load(Y)
    for i in range(0, N):
        x = tl.load(p).to(tl.float32) * scale
        y = tl.load(Y) + x
        tl.store(Y, y)
        p += 1
  }

def explicitLoads (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    p = X + 0
    __veritile_source_load_0 = tl.load(Y)
    for i in range(0, N):
        __veritile_source_load_1 = tl.load(p)
        x = __veritile_source_load_1.to(tl.float32) * scale
        __veritile_source_load_2 = tl.load(Y)
        y = __veritile_source_load_2 + x
        tl.store(Y, y)
        p += 1
  }


#print source
#print explicitLoads
end VeriTile.Tests.AliasedNestedLoadBody
