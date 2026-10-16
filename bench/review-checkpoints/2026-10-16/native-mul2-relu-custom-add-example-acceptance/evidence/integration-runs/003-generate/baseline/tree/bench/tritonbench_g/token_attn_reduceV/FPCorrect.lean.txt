import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPShape
import bench.tritonbench_g.token_attn_reduceV.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section

/-- Complete arbitrary-tripcount PV stream, including the empty stream and
partial final blocks. No finite-input, zero-product, reassociation or real-sum
law is assumed. The total reduction callback specifies the actual full tile. -/
specification fp32_stream_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s : State α)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s = some t ∧
      t.mem = (reference M R p s).mem ∧
      (∀ i : TileIndex [p.width], t.mem p.output (outputOffset p s i) =
        .mk .fp32 (outputValues M R p s i)) ∧
      (∀ r off, Outside p s r off → t.mem r off = s.mem r off) := by
  obtain ⟨t,he,hm⟩ := complete_run M R p s
  refine ⟨t,he,hm,?_,?_⟩
  · intro i
    rw [hm]
    exact reference_readback M R p s hinj i
  · intro r off h
    rw [hm]
    exact reference_frame M R p s r off h

/-- Original accepted launch domain, with arbitrary budget T and no extra
relationship between T and sequence length. The stronger execution theorem
also covers non-unit strides by retaining the source address formulas. -/
specification fp32_stream_io_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (R : SumPrimitive α) (p : Parameters) (s : State α) (T : Nat)
    (_hpbs : p.probTokenStride = 1) (_hrts : p.requestTokenStride = 1)
    (_hBN : 0 < p.block) (_hbudget : count p s ≤ T*p.block)
    (hinj : Function.Injective (outputOffset p s)) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], t.mem p.output (outputOffset p s i) =
        .mk .fp32 (outputValues M R p s i)) ∧
      (∀ r off, Outside p s r off → t.mem r off = s.mem r off) := by
  obtain ⟨t,he,_,hv,hf⟩ := fp32_stream_correctness M R p s hinj
  exact ⟨t,he,hv,hf⟩

/-- Closed software fp32 interpretation: every used numerical callback is
ScalarOps or the explicit ascending-lane sum. Native tree, fusion and FTZ/NaN
configuration correspondence is a separate obligation. -/
specification fp32_software_stream_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hinj : Function.Injective (outputOffset p s)) :
    ∃ t, FP.Structural.exec (withSum (software rest cfg) (softwareSum cfg)) (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], t.mem p.output (outputOffset p s i) =
        .mk .fp32 (bitOutput cfg p s i)) ∧
      (∀ r off, Outside p s r off → t.mem r off = s.mem r off) := by
  obtain ⟨t,he,_,hv,hf⟩ := fp32_stream_correctness (software rest cfg) (softwareSum cfg) p s hinj
  refine ⟨t,he,?_,hf⟩
  intro i
  simpa only [output_software] using hv i

#axiomsClean fp32_stream_correctness
#axiomsClean fp32_stream_io_correctness
#axiomsClean fp32_software_stream_correctness
end
end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPCorrect
