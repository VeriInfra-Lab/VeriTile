import bench.tritonbench_g.sin_kernel.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinKernel.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (n : Nat) : Program where
  kernel := originalKernel n
  domain := []
def indexCommuted (n : Nat) : Program where
  kernel := candidateKernel n
  domain := []

/-- Both actual kernels succeed and have equal complete typed memories. Only
integer indexing changes; no numerical law or finite-input guard is needed. -/
specification index_commuted_equiv (n : Nat) : original n ≡[assumptions] indexCommuted n := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  let result := Rewrite.originalExecution M n s
  exact ⟨result.val,result.val,result.property,
    (Rewrite.execution_eq M n s).symm.trans result.property,rfl⟩

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.FPEquiv
