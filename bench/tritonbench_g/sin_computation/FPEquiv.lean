import bench.tritonbench_g.sin_computation.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (X O : RegionName) (n : Nat) : Program where
  kernel := kernel X O n blockSize
  domain := []
def indexCommuted (X O : RegionName) (n : Nat) : Program where
  kernel := indexCommutedKernel X O n blockSize
  domain := []

/-- Both actual kernels succeed and have equal complete typed memories. Only
integer indexing changes; no numerical law or finite-input guard is needed. -/
specification index_commuted_equiv (X O : RegionName) (n : Nat) :
    original X O n ≡[assumptions] indexCommuted X O n := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  let result := Rewrite.originalExecution M X O n s
  exact ⟨result.val,result.val,result.property,
    (Rewrite.execution_eq M X O n s).symm.trans result.property,rfl⟩

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.FPEquiv
