import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.FastCeLoss.FPEquiv
open VeriTile.Triton KernelSource
open scoped VeriTile.Spec
noncomputable section

abbrev forwardOriginal := SourceEquivalence.forwardProgram Bool.false
abbrev forwardIndexCommuted := SourceEquivalence.forwardProgram Bool.true
abbrev chunkedOriginal := SourceEquivalence.chunkedProgram Bool.false
abbrev chunkedIndexCommuted := SourceEquivalence.chunkedProgram Bool.true
abbrev backwardOriginal := SourceEquivalence.backwardProgram Bool.false
abbrev backwardIndexCommuted := SourceEquivalence.backwardProgram Bool.true
def assumptions := SourceEquivalence.assumptions

/-- Complete repaired FP32 sources versus the retained signed row-index
commutation. Int64 labels, every mask, runtime branch, reduction and cast are
unchanged. The equality preserves complete states and failure behavior. -/
specification forward_kernel (p : Parameters) :
    forwardOriginal p ≡[assumptions] forwardIndexCommuted p :=
  SourceEquivalence.forward_equivalent p

specification chunked_kernel (p : Parameters) :
    chunkedOriginal p ≡[assumptions] chunkedIndexCommuted p :=
  SourceEquivalence.chunked_equivalent p

specification backward_kernel (p : Parameters) :
    backwardOriginal p ≡[assumptions] backwardIndexCommuted p :=
  SourceEquivalence.backward_equivalent p

#print_fp_assumptions forward_kernel
#print_fp_assumptions chunked_kernel
#print_fp_assumptions backward_kernel
#axiomsClean forward_kernel
#axiomsClean chunked_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.FPEquiv
