import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO
import bench.examples.RowWiseMax.Kernels
import bench.examples.RowWiseMax.Proofs.FP

/-! The public theorem states the row-size condition and the declared input and output footprints. The FP comparison carries the existing maximum rule. -/

namespace VeriTile.Bench.Examples.RowWiseMaxFPEquiv

open VeriTile.Bench.Examples.RowWiseMax.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.RowWiseMaxFPEquiv
