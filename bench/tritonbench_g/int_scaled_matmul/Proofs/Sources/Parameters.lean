import VeriTile.Triton.DSL

/-! Pure call parameters shared by both repaired source stages. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Triton
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner groupRows : Nat)
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
open VeriTile.Triton
structure Parameters where
  (input weight output scale : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride scaleRowStride scaleColumnStride
    blockRows blockColumns blockInner groupRows steps : Nat)
  (even : Bool)
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.ScaledWord
