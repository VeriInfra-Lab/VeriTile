import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPComplete
import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Complete fp16 source graph over the original accepted aligned layout and
arbitrary key-block count. All half casts/additions, explicit dot options,
fp32 online recurrence, and final half store remain observable operations.
The result is an independent initial-memory recurrence with every output and
full frame; no rounding identity, finite-domain restriction, or hstep premise.
Original native TTIR fuses dot/add operations and is a separate graph boundary. -/
specification attention_aligned_full_backend_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α) (p : Parameters)
    (initial : State α) (hBM : 0<p.blockM) (hBN : 0<p.blockN)
    (hHD : 0<p.head) (hnB : 1≤p.blocks) :
    ∃ final, FP.Structural.exec (backend rest P) (source p) initial=some final ∧
      final.mem=(written P p initial).mem ∧
      (∀ i : QLane p, final.mem p.output (qOffset p initial i)=
        .mk .fp16 (outputValue P p initial i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : QLane p, o≠qOffset p initial i) →
        final.mem r o=initial.mem r o) := correctness rest P p initial hBN hHD

/-- Actual binary16 conversions/addition and fp32 scalar operations, with
checked explicit reduction trees. Host scalar conversion, exp2, and typed dot
remain named primitive callbacks. This is neither a fully closed native model
nor a claim that separate dot/add equals the compiler's fused dot graph. -/
specification attention_aligned_full_scalar_bit_correctness
    [Inhabited Word]
    (rest : Algebra Word) (external : External) (plans : Plans) (p : Parameters)
    (initial : State Word) (hBM : 0<p.blockM) (hBN : 0<p.blockN)
    (hHD : 0<p.head) (hnB : 1≤p.blocks) :
    ∃ final, FP.Structural.exec (backend rest (software external plans)) (source p) initial=some final ∧
      final.mem=(written (software external plans) p initial).mem ∧
      (∀ i : QLane p, final.mem p.output (qOffset p initial i)=
        .mk .fp16 (outputValue (software external plans) p initial i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : QLane p, o≠qOffset p initial i) →
        final.mem r o=initial.mem r o) := correctness rest (software external plans) p initial hBN hHD

#axiomsClean attention_aligned_full_backend_correctness
#axiomsClean attention_aligned_full_scalar_bit_correctness
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPCorrect
