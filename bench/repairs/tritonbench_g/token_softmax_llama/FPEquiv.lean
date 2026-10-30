import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.KernelSource
import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.FPEquiv
open VeriTile Triton FP.ProfiledRewrite Strided
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Parameters) : Program where
  kernel := kernel .fp32 p
  domain := []
def indexCommuted (p : Parameters) : Program where
  kernel := KernelSource.candidateKernel p
  domain := []

/-- The complete FP32 source is equivalent after commuting its integer
length-address product. No floating transformation law is used.
The full native IO observation remains PublicIO.native; the previous combined
full-grid source/candidate endpoint remains in Misc/KernelNativeResults. -/
specification token_softmax_source_equiv (p : Parameters) :
    original p ≡[assumptions] indexCommuted p := KernelSource.equivalent p

#print_fp_assumptions token_softmax_source_equiv
#axiomsClean token_softmax_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.FPEquiv
