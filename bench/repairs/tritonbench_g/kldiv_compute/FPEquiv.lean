import bench.repairs.tritonbench_g.kldiv_compute.IO
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Equivalence

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec

abbrev original := FloatingIO.original
abbrev commuted := FloatingIO.commuted

/-- Address commutation leaves all floating operations in their original order. -/
def assumptions : Spec.Assumptions ComputeStmt := []

/-- Both kernels execute successfully, agree on all active outputs, and
preserve the exterior memory. The candidate swaps only `pid * B` to `B * pid`. -/
specification kldiv_equiv (p : Parameters) :
    original p ≡[assumptions] commuted p := by
  apply Spec.FloatingPoint.ofStructural (lhs := original p) (rhs := commuted p)
    (structural := MaskedIO₂Equiv) rfl rfl
  exact EquivalenceProof.equivalent p

#stmtSurfaceSubset kldiv_equiv ⊆
  [Parameters, MaskedKernelIO₂, original, commuted, assumptions, Spec.Equiv,
    maskedKernelIO₂FPProgramSyntax]
#print_fp_assumptions kldiv_equiv
#axiomsClean kldiv_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.Kernel.FPEquiv
