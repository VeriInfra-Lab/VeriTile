import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedSoftware

namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPCorrect
open VeriTile.Triton FP.Structural TypedFP
noncomputable section

/-- Full eleven-channel forward source, evaluated in source operation order.
The value specification reads only the initial coordinate memory. -/
specification spherical_forward_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α)
    (hstride : 10 < p.outputStride) (cover : InputCover p s) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ c i, outputOffset p s c i < p.outputNumel →
        t.mem p.output (outputOffset p s c i) = .mk .real (rawExpected M p s c i)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) :=
  source_correct M p s hstride cover

/-- Every arithmetic operation uses the binary32 bit interpreter. Decimal
coefficient encodings are checked by `coefficient_words`; multiplication and
addition are separate operations in this profile. -/
specification spherical_forward_software_fp32 [Inhabited Word]
    (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hstride : 10 < p.outputStride) (cover : InputCover p s) :
    ∃ t, FP.Structural.exec (software rest cfg) (kernel p) s = some t ∧
      (∀ c i, outputOffset p s c i < p.outputNumel →
        t.mem p.output (outputOffset p s c i) = .mk .real
          (bitChannels cfg ((s.mem p.input (coordOffset p s i)).read .real)
            ((s.mem p.input (coordOffset p s i + 1)).read .real)
            ((s.mem p.input (coordOffset p s i + 2)).read .real) c)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := source_correct (software rest cfg) p s hstride cover
  refine ⟨t, he, ?_, hf⟩
  intro c i hi
  have value := hv c i hi
  dsimp only [rawExpected] at value
  rw [channels_software] at value
  exact value

#axiomsClean spherical_forward_typed_correctness
#axiomsClean spherical_forward_software_fp32
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPCorrect
