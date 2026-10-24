import bench.tritonbench_g.attention_kernel_aligned.Proofs.FullSourceBehavior
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BfloatRejected
import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullCorrect
open VeriTile.Triton FP.Structural
open Full
open FullFp16
open FPValue (Primitives backend Word External Plans software)
noncomputable section

/-- Original fp16 source with arbitrary N_CTX and every individual source
stride. Full final key/query tiles remain unmasked, including out-of-shape
lanes. Every key iteration and the entire final memory scatter are accounted
for, even when output strides collide. This total-memory source contract does
not assert allocation safety or native execution. The bf16 wrapper branch is
rejected by both the unchanged frontend and the recorded native compiler;
the imported full-source negative regression checks that rejection. -/
specification attention_aligned_all_lengths_backend_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (hBN : 0 < p.blockN) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial = some final ∧
      final.mem = (written P p initial).mem ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : QLane p, o ≠ outputOffset p initial i) →
        final.mem r o = initial.mem r o) := correctness rest P p initial hBN

/-- Binary16 conversions/addition and fp32 scalar arithmetic instantiated on
the same complete source. Valid explicit reduction trees remain checked; dot,
exp2, and arbitrary-real host literal conversion are named external callbacks.
No identity-rounding, alignment, finite-value, or step-correctness premise is
introduced. Native compilation, unchecked memory safety, and host grid
composition remain separate from this per-program value contract. -/
specification attention_aligned_all_lengths_scalar_bit_correctness
    [Inhabited Word] (rest : Algebra Word) (external : External) (plans : Plans)
    (p : Parameters) (initial : State Word) (hBN : 0 < p.blockN) :
    ∃ final, FP.Structural.exec (backend rest (software external plans)) (source p) initial = some final ∧
      final.mem = (written (software external plans) p initial).mem ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : QLane p, o ≠ outputOffset p initial i) →
        final.mem r o = initial.mem r o) := correctness rest (software external plans) p initial hBN

#axiomsClean attention_aligned_all_lengths_backend_correctness
#axiomsClean attention_aligned_all_lengths_scalar_bit_correctness
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullCorrect
