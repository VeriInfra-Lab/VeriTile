import VeriTile.Triton
/-! Shared call metadata for the literal-parameter RmsRbeMatmul source. The
separate copied kernel family has been retired; executable source lives in
rms_rbe_matmul/Kernels.lean. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource
open VeriTile.Triton
structure Arguments where
  (X W RMS OUT : RegionName)
  (M N K sxb sxm sxk swk swn srms sob som son : Nat)
  (BM BN BK headDim : Nat)
  start : Int
  rotary : Bool
  (theta epsilon : ℝ)

end VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource
