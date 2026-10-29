import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedFrameTail
import bench.tritonbench_g.layer_norm_fwd.Proofs.InputDomain

/-! Actual complete typed source success and exact active-store frame. No
numeric law, output formula, source/native equality or physical codec law is
assumed. Output readback and native lowering are separate open obligations. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 5000

theorem written_is_writeRegion
    (h : Contract.written f r stride M N row region offset) : Domain.writeRegion f r region := by
  unfold Contract.written at h
  unfold Domain.writeRegion
  tauto

theorem Frame.inputs_unchanged {α : Type} {s t : State α}
    (h : Frame f r stride M N row s t) (hsep : Domain.ReadWriteSeparated f r)
    (region : RegionName) (hr : Domain.readRegion f r region) (offset : Nat) :
    t.mem region offset=s.mem region offset := by
  apply h region offset
  intro hw
  exact hsep region hr region (written_is_writeRegion hw) rfl

theorem source_success_frame {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α) :
    ∃ t, FP.Structural.exec (engine A) (repaired f r stride M N eps p block hb) s=some t ∧
      Frame f r stride M N (s.pids 0) s t := by
  obtain ⟨s₁,h₁,hr₁⟩ := prefix_success A f r stride M N eps p block hb s
  obtain ⟨s₂,h₂,hr₂⟩ := firstStream_success A cmp f r stride M N eps p block hb (s.pids 0) s₁ hr₁
  obtain ⟨s₃,h₃,hr₃⟩ := secondInput_success A cmp f r stride M N eps p block hb (s.pids 0) s₂ hr₂
  obtain ⟨s₄,h₄,hr₄⟩ := residual_success A f r stride M N eps p block hb (s.pids 0) s₃ hr₃
  obtain ⟨s₅,h₅,hr₅⟩ := normalization_success A f r stride M N eps p block hb (s.pids 0) s₄ hr₄
  obtain ⟨s₆,h₆⟩ := affine_success A f r stride M N eps p block hb (s.pids 0) s₅ hr₅
  have hf₁ : Frame f r stride M N (s.pids 0) s s₁ := by
    intro region offset _
    rw [prefix_frame A f r stride M N eps p block hb s s₁ h₁]
  have hf₂ := firstStream_frame A cmp f r stride M N eps p block hb (s.pids 0) s₁ s₂ hr₁ h₂
  have hf₃ := secondInput_frame A cmp f r stride M N eps p block hb (s.pids 0) s₂ s₃ hr₂ h₃
  have hf₄ := residual_frame A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ h₄
  have hf₅ := normalization_frame A f r stride M N eps p block hb (s.pids 0) s₄ s₅ hr₄ h₅
  have hf₆ := affine_frame A f r stride M N eps p block hb (s.pids 0) s₅ s₆ hr₅ h₆
  refine ⟨s₆,?_,hf₁.trans (hf₂.trans (hf₃.trans (hf₄.trans (hf₅.trans hf₆))))⟩
  let code := (repaired f r stride M N eps p block hb).surfaceBody
  change run (engine A) (code.take 10) s = some s₁ at h₁
  change run (engine A) ((code.drop 10).take 2) s₁ = some s₂ at h₂
  change run (engine A) ((code.drop 12).take 1) s₂ = some s₃ at h₃
  change run (engine A) ((code.drop 13).take 2) s₃ = some s₄ at h₄
  change run (engine A) ((code.drop 15).take 3) s₄ = some s₅ at h₅
  change run (engine A) (code.drop 18) s₅ = some s₆ at h₆
  change run (engine A) code s = some s₆
  rw [← List.take_append_drop 10 code,run_append,h₁,Option.bind_some]
  rw [← List.take_append_drop 2 (code.drop 10),run_append,h₂,Option.bind_some]
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [← List.take_append_drop 1 (code.drop 12),run_append,h₃,Option.bind_some]
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [← List.take_append_drop 2 (code.drop 13),run_append,h₄,Option.bind_some]
  simp only [List.drop_drop,Nat.reduceAdd]
  rw [← List.take_append_drop 3 (code.drop 15),run_append,h₅,Option.bind_some]
  simpa only [List.drop_drop,Nat.reduceAdd] using h₆

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
