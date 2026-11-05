import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.RewriteTactics
import bench.examples.TritonBenchVectorAddition.Kernels

/-! The launch takes the logical element count and block size. The source masks reads and writes to the logical count. -/

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv

open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels

open VeriTile Triton FP.Structural FP.GuardedRewrite

open scoped VeriTile.Spec

abbrev Rules := FP.ScalarArithmetic.Rules

end VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv
