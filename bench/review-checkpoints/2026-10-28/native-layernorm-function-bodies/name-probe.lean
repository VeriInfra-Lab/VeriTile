import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.NativeIO
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RealPrimitives
set_option pp.fullNames true
namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
#check @TypedSource.cast
#check @TypedSource.meanValue
#check cast
#check default_mul
#check algebra
open VeriTile.Bench.Repairs.LayerNormFwd
#check @TypedSource.cast
#check @TypedSource.meanValue
#check cast
#check default_mul
#check algebra
example (x : ℝ) : TypedSource.cast algebra x = x := by exact cast x
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
