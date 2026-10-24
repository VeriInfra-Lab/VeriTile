import bench.tritonbench_g.layer_norm_liger.Proofs.FPContract
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSupplied
import bench.tritonbench_g.layer_norm_liger.Proofs.FPShape
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPCorrect
open VeriTile.Triton FP.Structural FPStages
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum softwareSum)
noncomputable section

/-- Complete fp32 forward with all three outputs. The full padded block feeds
the variance reduction; no finite-input, associativity or exact-rounding law
is required. Output separation has the same role as in the old forward scope. -/
specification fp32_forward_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (R : SumPrimitive α) (p : Parameters) (s : State α)
    (hmy : p.mean ≠ p.y) (hmr : p.mean ≠ p.rstd) (hry : p.rstd ≠ p.y) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s = some t ∧
      t.mem = (reference M R p s).mem ∧
      (∀ i : TileIndex [p.block], i.1.val < p.columns →
        t.mem p.y (outputOffset p s i) = stored M (yValue M R p s i)) ∧
      t.mem p.mean (meanOffset p s) = stored M (meanValue M R p s) ∧
      t.mem p.rstd (rstdOffset p s) = stored M (rstdValue M R p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t,he,hm⟩ := complete_run M R p s
  refine ⟨t,he,hm,?_,?_,?_,?_⟩
  · intro i hi
    rw [hm]
    exact reference_y M R p s i hi
  · rw [hm]
    exact reference_mean M R p s hmy hmr
  · rw [hm]
    exact reference_rstd M R p s hry
  · intro r o h
    rw [hm]
    exact reference_frame M R p s r o h

/-- ScalarOps and an explicit full-lane reduction plan. The literal conversion
and rsqrt instruction are named inputs, so this is a backend value contract,
not a closed rsqrt bit interpreter or a GPU conformance claim. -/
specification fp32_scalarops_forward_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (literal : ℝ → FP.Value .fp32) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (p : Parameters) (s : State (FP.Value .fp32))
    (hmy : p.mean ≠ p.y) (hmr : p.mean ≠ p.rstd) (hry : p.rstd ≠ p.y) :
    ∃ t, FP.Structural.exec (withSum (software rest cfg literal rsqrt32) (softwareSum cfg))
      (kernel p) s = some t ∧
      (∀ i : TileIndex [p.block], i.1.val < p.columns →
        t.mem p.y (outputOffset p s i) = .mk .fp32 (bitY cfg literal rsqrt32 p s i)) ∧
      t.mem p.mean (meanOffset p s) = .mk .fp32 (bitMean cfg literal p s) ∧
      t.mem p.rstd (rstdOffset p s) = .mk .fp32 (bitRstd cfg literal rsqrt32 p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  obtain ⟨t,he,_,hy,hm,hr,hf⟩ := fp32_forward_correctness
    (software rest cfg literal rsqrt32) (softwareSum cfg) p s hmy hmr hry
  have hv := software_values rest cfg literal rsqrt32 p s
  exact ⟨t,he,fun i hi => (hy i hi).trans (hv.2.2 i),hm.trans hv.1,hr.trans hv.2.1,hf⟩

/-- A supplied MeanPre scalar is independently copied; it is not asserted to
be the mean computed by the complete forward stage. -/
specification supplied_mean_store_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (pre output : RegionName) (stride : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (meanStore .fp32 pre output stride) s = some t ∧
      t.mem output (s.pids 0*stride) = .mk .fp32 ((s.mem pre (s.pids 0*stride)).read .fp32) ∧
      (∀ r o, (r ≠ output ∨ o ≠ s.pids 0*stride) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := mean_store_run M pre output stride s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    simp only [State.write]
    have : ¬(r=output ∧ o=s.pids 0*stride) := by tauto
    simp [this]

/-- The accepted independent supplied-RSTD store stage. -/
specification supplied_rstd_store_correctness {α : Type} [Inhabited α] (M : Algebra α)
    (pre output : RegionName) (stride : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (rstdStore .fp32 pre output stride) s = some t ∧
      t.mem output (s.pids 0*stride) = .mk .fp32 ((s.mem pre (s.pids 0*stride)).read .fp32) ∧
      (∀ r o, (r ≠ output ∨ o ≠ s.pids 0*stride) → t.mem r o=s.mem r o) := by
  obtain ⟨t,he,hm⟩ := rstd_store_run M pre output stride s
  refine ⟨t,he,?_,?_⟩
  · rw [hm]; exact State.write_same _ _ _ _
  · intro r o h
    rw [hm]
    simp only [State.write]
    have : ¬(r=output ∧ o=s.pids 0*stride) := by tauto
    simp [this]

#axiomsClean fp32_forward_correctness
#axiomsClean fp32_scalarops_forward_correctness
#axiomsClean supplied_mean_store_correctness
#axiomsClean supplied_rstd_store_correctness
end
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPCorrect
