import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.AdamUpdateGridLaunch.Kernels

/-! The launch uses the supplied block size and logical element count. The source masks lanes beyond that count and reads and writes the parameter and momentum arrays in place. -/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv

open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv
