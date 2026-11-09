import VeriTile.Triton.DSL.PythonInlineSuites

namespace VeriTile.Tests.StaticRangesReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("Out",.i32)]
  constexpr := ["HEAD_DIM","KCACHE_X"]

def reference (Out : RegionName) (HEAD_DIM KCACHE_X : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    for split_x in range($(HEAD_DIM / KCACHE_X)) {
      offsets = split_x * KCACHE_X + tl.arange(0, KCACHE_X)
      tl.store(Out + offsets, offsets)
    }
  }
end VeriTile.Tests.StaticRangesReference
