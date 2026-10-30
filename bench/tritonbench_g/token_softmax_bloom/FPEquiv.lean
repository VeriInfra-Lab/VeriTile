import bench.tritonbench_g.token_softmax_bloom.Kernels
import bench.tritonbench_g.token_softmax_bloom.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Parameters) : Program where
  kernel := kernel .fp32 p
  domain := []
def indexCommuted (p : Parameters) : Program where
  kernel := KernelRewrite.candidateKernel p
  domain := []

/-- The complete selected FP32 source is equivalent after commuting its one
integer metadata-address product. No floating transformation law is used. -/
specification token_softmax_source_equiv (p : Parameters) :
    original p ≡[assumptions] indexCommuted p := KernelRewrite.equivalent p

#print_fp_assumptions token_softmax_source_equiv
#axiomsClean token_softmax_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.FPEquiv
