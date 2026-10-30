import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelNativeResults
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheCopy.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Source.Parameters) : Program := ⟨Source.originalKernel p,[]⟩
def commuted (p : Source.Parameters) : Program := ⟨Source.commutedKernel p,[]⟩

/-- Both complete static-loop kernels succeed and have identical full typed
memory. Copy payloads are unrestricted; no numerical transformation law. -/
specification source_equivalence (p : Source.Parameters) :
    original p ≡[assumptions] commuted p :=
  KernelNativeResults.source_equivalence p

#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.FPEquiv
