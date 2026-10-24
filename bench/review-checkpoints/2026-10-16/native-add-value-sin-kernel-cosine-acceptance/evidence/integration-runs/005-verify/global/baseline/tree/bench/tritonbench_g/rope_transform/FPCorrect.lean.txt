import bench.tritonbench_g.rope_transform.Proofs.TypedShape
import bench.tritonbench_g.rope_transform.Proofs.TypedWitness

/-! Full fp32 rotary source value contracts. The coefficient buffers are inputs;
all four Q/K stores and both branches execute. Native compiler fusion, rounding
configuration and allocation/grid correspondence require a separate binding. -/
namespace VeriTile.Bench.TritonBenchG.RopeTransform.FPCorrect
open VeriTile.Triton FP.Structural Typed
noncomputable section

specification rope_fp32_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (kernel .fp32 .fp32 p) s = some t ∧
      t.mem = referenceMemory M .fp32 .fp32 p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  refine ⟨(execution M p s).val, (execution M p s).property, execution_memory M p s, ?_⟩
  intro r o h
  rw [execution_memory]
  exact reference_frame M .fp32 .fp32 p s r o h

/-- Every output uses separately rounded binary32 products and add/subtract.
The independent reference contains no callback from the surrounding backend. -/
specification rope_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (kernel .fp32 .fp32 p) s = some t ∧
      t.mem = roundedMemory cfg p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm, hf⟩ := rope_fp32_correctness (FP.Software.scalarFP32 rest cfg) p s
  exact ⟨t, he, hm.trans (reference_software rest cfg p s), hf⟩

theorem rope_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (kernel .fp32 .fp32 witnessParameters) witnessState = some t ∧
      t.mem "Q" 0 = .mk .fp32 witnessOne ∧ t.mem "Q" 1 = .mk .fp32 witnessOne ∧
      t.mem "K" 0 = .mk .fp32 witnessOne ∧ t.mem "K" 1 = .mk .fp32 witnessOne ∧
      t.mem "Q" 2 = .mk .fp32 witnessSmall ∧ (1 - 1 / 33554432 : Rat) ≠ 1 :=
  bit_success rest

#axiomsClean rope_fp32_correctness
#axiomsClean rope_software_fp32
#axiomsClean rope_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.RopeTransform.FPCorrect
