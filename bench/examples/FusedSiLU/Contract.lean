import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO
import bench.examples.FusedSiLU.Kernels
import bench.examples.FusedSiLU.Proofs.FP

/-! The input uses the declared tensor layout and dtype. The FP theorem names the numerical rule for the unfused and fused comparison. -/

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv

open VeriTile.Bench.Examples.FusedSiLU.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

/-- Fusion preserves the same opaque numerical call tree, so no numerical
identity is assumed in this example. -/
def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
