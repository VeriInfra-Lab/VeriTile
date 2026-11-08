import bench.examples.FusedSiLU.IO
import bench.examples.FusedSiLU.Contract
import bench.examples.FusedSiLU.Kernels
import bench.examples.FusedSiLU.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.FusedSiLUFPEquiv
open VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

open scoped VeriTile.Spec

specification silu_equiv (B : Nat) : fusedIO B ≡[R] unfusedIO B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO B) (rhs := unfusedIO B)
    (structural := IO₃Equiv) rfl rfl
  exact Execution.equivalent B

#print_fp_assumptions silu_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSiLUFPEquiv
