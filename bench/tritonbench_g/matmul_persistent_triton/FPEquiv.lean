import bench.tritonbench_g.matmul_persistent_triton.Kernels
import bench.tritonbench_g.matmul_persistent_triton.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def program (p : Parameters) (storage : TypedSource.Storage) (v : Bool) : Program where
  kernel := MatmulPersistentTriton.kernel p storage v
  domain := []

/-- Exact complete typed-source equivalence under every backend algebra for
both original wrapper storage profiles. The sole rewrite is an exact signed
integer address multiplication; all floating computations, casts, persistent
loop iterations, masks, stores, resets, failures and full states are retained.
No floating numerical assumption or successful-execution premise is used.
This is the formal typed source; native fixed-width integer correspondence
and legal tile/dtype/launch admission remain separate obligations. -/
specification source_equivalence (p : Parameters) (storage : TypedSource.Storage) :
    program p storage Bool.false ≡[assumptions] program p storage Bool.true :=
  FPProof.source_equivalence p storage

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FPEquiv
