import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.FP32ReferenceRewrite
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.HyperConnectionsWidth.Kernels

/-! The public statements supply the matrix dimensions, iteration count and temperature. The fixed-rank and general matrix theorems expose their respective input conditions. -/

namespace VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv

open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv
