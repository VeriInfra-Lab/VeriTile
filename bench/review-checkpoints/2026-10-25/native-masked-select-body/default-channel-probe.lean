import bench.tests.CallSignature
namespace VeriTile.Tests.CallSignature.NaturalIntegerMemory.DefaultChannelProbe
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("P",.i64),("Y",.fp32)]
  constexpr := ["B"]

def source (X Y : RegionName) (P : Region .nat) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lanes = tl.arange(0, B)
    values = tl.load(X + lanes)
    prefix_value = tl.load(P + lanes)
    indices = prefix_value - 1
    tl.store(Y + indices, values)
  }

/-- Expected rejection: declared int64 storage alone does not select the
caller's natural-valued mathematical IO channel or prove any range bound. -/
theorem without_explicit_projection (X Y : RegionName) (P : Region .nat) (B : Nat) :
    source X Y P B = NaturalIntegerMemory.source X Y P B := rfl
end VeriTile.Tests.CallSignature.NaturalIntegerMemory.DefaultChannelProbe
