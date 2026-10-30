import bench.examples.FusedSwiglu.Kernels
import bench.examples.FusedSwiglu.Proofs.FP
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/-! FP equivalence of the sources in Kernels.lean.
The IO contracts and required numerical domains are stated below; detailed
execution and scalar derivations are in Proofs/FP.lean. -/

namespace VeriTile.Bench.Examples.FusedSwigluFPEquiv
open VeriTile.Bench.Examples.FusedSwiglu.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

abbrev unfusedIO := FusedSwiglu.IO.unfusedIO
abbrev fusedIO := FusedSwiglu.IO.fusedIO

def R : Spec.Assumptions ComputeStmt := []

open scoped VeriTile.Spec

/-- Both original kernels compute the same numerical expression on active
lanes. Dimensions are arbitrary, including zero and partial blocks. -/
specification swiglu_equiv (n B : Nat) : fusedIO n B ≡[R] unfusedIO n B := by
  apply Spec.FloatingPoint.ofStructural (lhs := fusedIO n B) (rhs := unfusedIO n B)
    (structural := MaskedIO₂Equiv) rfl rfl
  exact Execution.equivalent n B

#print_fp_assumptions swiglu_equiv

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSwigluFPEquiv
