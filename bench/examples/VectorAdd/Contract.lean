import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.VectorAdd.Kernels

/-! The launch supplies a block size. The Real and FP interfaces give the aligned input and output footprints. -/

namespace VeriTile.Bench.Examples.VectorAddFPEquiv

open VeriTile.Bench.Examples.VectorAdd.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.VectorAddFPEquiv
