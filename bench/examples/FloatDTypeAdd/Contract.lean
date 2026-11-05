import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.FloatDTypeAdd.Kernels

/-! The tile length is a natural number, including zero. The typed source uses FP32 storage and an explicit output conversion. -/

namespace VeriTile.Bench.Examples.FloatDTypeAddFPEquiv

open VeriTile.Bench.Examples.FloatDTypeAdd.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.FloatDTypeAddFPEquiv
