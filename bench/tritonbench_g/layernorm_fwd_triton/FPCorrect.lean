import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPContract
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPShape
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section

/-- Complete three-pass streaming forward. All N outputs are characterized
for arbitrary positive block size, with original input/output disjointness.
Every tail operation remains present, including W's address-indexed undef
load and arithmetic on lanes subsequently masked at store. -/
specification fp32_full_stream_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α)
    (hb : 0<p.block) (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) :
    ∃ t, FP.Structural.exec M (kernel p) s=some t ∧
      t.mem=referenceMem M p s ∧
      (∀ n, n<p.columns → t.mem p.y (yBase p s+n)=.mk .fp32 (yValue M p s n)) ∧
      (∀ r o, Outside p s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := complete_run M p s hx hw
  refine ⟨t,he,hm,?_,?_⟩
  · intro n hn
    rw [hm]
    exact reference_readback M p s hb n hn
  · intro r o h
    rw [hm]
    exact reference_frame M p s r o h

/-- Complete ScalarOps interpretation with an explicit reduction plan; sqrt
and source-literal conversion are explicit native-primitive bindings. The
result is independent of all unused callbacks in rest, but does not establish
GPU sqrt, fusion, reduction-tree or launch conformance. -/
specification fp32_scalarops_full_stream_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (sqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32))
    (hb : 0<p.block) (hx : p.x ≠ p.y) (hw : p.weight ≠ p.y) :
    ∃ t, FP.Structural.exec (software rest cfg literal sqrt32) (kernel p) s=some t ∧
      (∀ n, n<p.columns → t.mem p.y (yBase p s+n)=.mk .fp32 (bitY cfg literal sqrt32 p s n)) ∧
      (∀ r o, Outside p s r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,_,hv,hf⟩ := fp32_full_stream_correctness
    (software rest cfg literal sqrt32) p s hb hx hw
  refine ⟨t,he,?_,hf⟩
  intro n hn
  simpa only [output_software] using hv n hn

#axiomsClean fp32_full_stream_correctness
#axiomsClean fp32_scalarops_full_stream_correctness
end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPCorrect
