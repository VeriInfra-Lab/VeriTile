import bench.repairs.tritonbench_g.embedding_triton_kernel.Kernels
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Rewrite
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Kernel

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.SuccessfulEquivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

abbrev assumptions := Rewrite.assumptions
abbrev original := Rewrite.original
abbrev indexCommuted := Rewrite.indexCommuted

/-- Actual original-versus-index-commuted equivalence of the shared typed
FP32/int32 source pair. It uses exact integer commutativity, no floating law.
All masks, load defaults, payload precision and loop iterations are retained. -/
theorem embedding_index_commuted_equiv (p : Parameters) :
    original p ≡[assumptions] indexCommuted p := Rewrite.equivalent p

/-- Both actual sources terminate under the copy/zero model and the declared
kernel memory contract, with identical complete typed memories. FP32 words
are arbitrary, including NaN payloads, infinities, subnormals and signed zero. -/
theorem embedding_kernel_successful_equivalence
    (rest : FP.Structural.Algebra (BitVec 32)) (p : Parameters)
    (bounds : RegionBounds) (s : FP.Structural.State (BitVec 32))
    (h : KernelContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    ∃ left right,
      FP.Structural.exec (Execution.backend rest 0) (original p).kernel s = some left ∧
      FP.Structural.exec (Execution.backend rest 0) (indexCommuted p).kernel s = some right ∧
      left.mem = right.mem ∧
      (∀ a,a < 64 → s.pids 0*64+a < p.n → ∀ d,d < p.hidden →
        left.mem p.output ((s.pids 0*64+a)*p.outputStride+d) =
          .mk .real (KernelProof.expected 0 p s (s.pids 0*64+a) d)) ∧
      (∀ r off,r ≠ p.output ∨ ¬KernelProof.Written p (s.pids 0) off →
        left.mem r off = s.mem r off) := by
  obtain ⟨t,run,values,frame⟩ := KernelProof.correct rest 0 p bounds s h
  refine ⟨t,t,run,?_,rfl,values,frame⟩
  change FP.Structural.exec (Execution.backend rest 0) (indexCommutedSource .int32 p) s = some t
  rw [←Rewrite.execution_eq]
  exact run

theorem original_source (p : Parameters) :
    (original p).kernel = source .int32 p := rfl

theorem candidate_source (p : Parameters) :
    (indexCommuted p).kernel = indexCommutedSource .int32 p := rfl

#print_fp_assumptions embedding_index_commuted_equiv
#axiomsClean embedding_index_commuted_equiv
#axiomsClean embedding_kernel_successful_equivalence
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.SuccessfulEquivalence
