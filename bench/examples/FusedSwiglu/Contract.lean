import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Kernel
import VeriTile.Triton.Float.StructuralIO
import bench.examples.FusedSwiglu.Kernels
import bench.examples.FusedSwiglu.Proofs.FP

/-! The two inputs use the declared tensor layout and dtype. The public FP theorem supplies the numerical rule for the activation comparison. -/

namespace VeriTile.Bench.Examples.FusedSwigluFPEquiv

open VeriTile.Bench.Examples.FusedSwiglu.Kernels

open VeriTile Triton

open FP.Structural

set_option maxHeartbeats 1200000

def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

end VeriTile.Bench.Examples.FusedSwigluFPEquiv
