import bench.repairs.tritonbench_g.embedding_triton_kernel.Kernels
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Kernel

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section

/-- Arbitrary-Real gather/zero correctness for the authoritative FP32 typed
source. The memory contract states the program's initial arrays, active bounds
and disjoint output role. No Real values are restricted to an encoder image. -/
theorem embedding_kernel_correctness (rest : Algebra ℝ) (p : Parameters)
    (bounds : RegionBounds) (s : State ℝ)
    (h : KernelContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    Spec.Real (∃ t,FP.Structural.exec (Execution.backend rest 0) (source .int32 p) s = some t ∧
      (∀ a,a < 64 → s.pids 0*64+a < p.n → ∀ d,d < p.hidden →
        t.mem p.output ((s.pids 0*64+a)*p.outputStride+d) =
          .mk .real (KernelProof.expected 0 p s (s.pids 0*64+a) d)) ∧
      (∀ r off,r ≠ p.output ∨ ¬KernelProof.Written p (s.pids 0) off →
        t.mem r off = s.mem r off)) :=
  KernelProof.correct rest 0 p bounds s h


export KernelProof (accesses)
#specNonCircular KernelProof.expected avoiding [source,indexCommutedSource,FP.Structural.exec]
#axiomsClean embedding_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.KernelRealResult
