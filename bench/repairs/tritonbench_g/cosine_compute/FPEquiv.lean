import bench.repairs.tritonbench_g.cosine_compute.Proofs.ReaderFP
import bench.repairs.tritonbench_g.cosine_compute.Proofs.Rewrite
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CosineCompute.FPEquiv
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
specification index_commuted_equiv (n : Nat) : original n ≡[assumptions] indexCommuted n :=
  ReaderProof.index_commuted_equiv n

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.FPEquiv
