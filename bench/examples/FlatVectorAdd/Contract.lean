import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.FlatVectorAdd.Kernels

/-! The launch uses the logical element count and block size. Loads and stores are masked to that count. The public Real and FP entries specify their memory layout and masks. -/

namespace VeriTile.Bench.Examples.FlatVectorAddFPEquiv

open VeriTile.Bench.Examples.FlatVectorAdd.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.FlatVectorAddFPEquiv
