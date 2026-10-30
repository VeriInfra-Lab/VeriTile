import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedStreamValues
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedWhole

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def ReadsEqual {α : Type} (f : Flags) (r : Regions) (current initial : State α) : Prop :=
  ∀ region, Domain.readRegion f r region → ∀ offset,
    current.mem region offset=initial.mem region offset

theorem Frame.readsEqual {α : Type} {s t : State α}
    (h : Frame f r stride M N row s t) (hsep : Domain.ReadWriteSeparated f r) :
    ReadsEqual f r t s := h.inputs_unchanged hsep

theorem streamValues_reads {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (s initial : State α) (h : ReadsEqual f r s initial)
    (index : Nat) (p : ℝ) (v : Values α .real [block]) :
    streamValues A cmp f s r index p v=streamValues A cmp f initial r index p v := by
  have hscale : f.rowscale=Bool.true → s.mem r.rowscale index=initial.mem r.rowscale index := by
    intro hf; exact h r.rowscale (by simp [Domain.readRegion,hf]) index
  have hseed : f.dropout=Bool.true → s.mem r.seeds index=initial.mem r.seeds index := by
    intro hf; exact h r.seeds (by simp [Domain.readRegion,hf]) index
  funext j
  cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [streamValues,scalarLoad,hs,hd,hscale,hseed]

def secondCombined {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (s : State α) (M N row : Nat)
    (p : ℝ) (v : Values α .real [block]) : Values α .real [block] := fun j =>
  if f.x1 then op A .add (v j) (streamValues A cmp f s r (M+row) p
    (fun k => raw A s r.x1 (row*stride.x1) N k.1.val) j) else v j

theorem secondCombined_reads {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (s initial : State α) (h : ReadsEqual f r s initial)
    (M N row : Nat) (p : ℝ) (v : Values α .real [block]) :
    secondCombined A cmp f r stride s M N row p v=secondCombined A cmp f r stride initial M N row p v := by
  have hx : f.x1=Bool.true → ∀ col, s.mem r.x1 (row*stride.x1+col)=initial.mem r.x1 (row*stride.x1+col) := by
    intro hf col; exact h r.x1 (by simp [Domain.readRegion,hf]) _
  cases hf : f.x1
  · funext j
    simp [secondCombined,hf]
  · have hv : (fun k : TileIndex [block] => raw A s r.x1 (row*stride.x1) N k.1.val)=
        (fun k => raw A initial r.x1 (row*stride.x1) N k.1.val) := by
      funext k; simp [raw,hx hf]
    unfold secondCombined
    rw [hv,streamValues_reads A cmp f r s initial h]

def residualCombined {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (s : State α) (N row : Nat)
    (v : Values α .real [block]) : Values α .real [block] := fun j =>
  if f.residual then op A .add (v j) (raw A s r.residual (row*stride.residual) N j.1.val) else v j

theorem residualCombined_reads {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (s initial : State α) (h : ReadsEqual f r s initial)
    (N row : Nat) (v : Values α .real [block]) :
    residualCombined A f r stride s N row v=residualCombined A f r stride initial N row v := by
  have hr : f.residual=Bool.true → ∀ col,
      s.mem r.residual (row*stride.residual+col)=initial.mem r.residual (row*stride.residual+col) := by
    intro hf col; exact h r.residual (by simp [Domain.readRegion,hf]) _
  funext j
  cases hf : f.residual <;> simp [residualCombined,raw,hf,hr]

theorem prefix_value_of_run {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s t : State α)
    (ht : run (engine A) (stagePrefix f r stride M N eps p block hb) s=some t) :
    t.regs .real [block] "x"=some (fun j => raw A s r.x ((s.pids 0)*stride.x) N j.1.val) := by
  obtain ⟨u,hu,hv⟩ := prefix_value A f r stride M N eps p block hb s
  have he : u=t := Option.some.inj (hu.symm.trans ht)
  simpa only [he] using hv

/-- Complete actual input prefix, including both mask stores and residual-out
store, computes its opaque FP values from the pre-prefix input memory. -/
theorem input_prefix_values {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) :
    ∃ t, run (engine A) ((repaired f r stride M N eps p block hb).surfaceBody.take 15) s=some t ∧
      Ready f r stride block (s.pids 0) t ∧ Frame f r stride M N (s.pids 0) s t ∧
      t.regs .real [block] "x"=some (combinedValues A cmp f s r stride M N p (s.pids 0) block) := by
  obtain ⟨s₁,h₁,hr₁⟩ := prefix_success A f r stride M N eps p block hb s
  obtain ⟨s₂,h₂,hr₂⟩ := firstStream_success A cmp f r stride M N eps p block hb (s.pids 0) s₁ hr₁
  obtain ⟨s₃,h₃,hr₃⟩ := secondInput_success A cmp f r stride M N eps p block hb (s.pids 0) s₂ hr₂
  obtain ⟨s₄,h₄,hr₄⟩ := residual_success A f r stride M N eps p block hb (s.pids 0) s₃ hr₃
  have hf₁ : Frame f r stride M N (s.pids 0) s s₁ := by
    intro region offset _
    rw [prefix_frame A f r stride M N eps p block hb s s₁ h₁]
  have hf₂ := firstStream_frame A cmp f r stride M N eps p block hb (s.pids 0) s₁ s₂ hr₁ h₂
  have hf₃ := secondInput_frame A cmp f r stride M N eps p block hb (s.pids 0) s₂ s₃ hr₂ h₃
  have hf₄ := residual_frame A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ h₄
  have hv₁ := prefix_value_of_run A f r stride M N eps p block hb s s₁ h₁
  have hv₂ := firstStream_value A cmp f r stride M N eps p block hb (s.pids 0) s₁ s₂ hr₁ _ hv₁ h₂
  rw [streamValues_reads A cmp f r s₁ s (hf₁.readsEqual hsep)] at hv₂
  have hv₃ := secondInput_value A cmp f r stride M N eps p block hb (s.pids 0) s₂ s₃ hr₂ _ hv₂ h₃
  change s₃.regs .real [block] "x"=some (secondCombined A cmp f r stride s₂ M N (s.pids 0) p _) at hv₃
  rw [secondCombined_reads A cmp f r stride s₂ s ((hf₁.trans hf₂).readsEqual hsep)] at hv₃
  have hv₄ := residual_value A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ _ hv₃ h₄
  change s₄.regs .real [block] "x"=some (residualCombined A f r stride s₃ N (s.pids 0) _) at hv₄
  rw [residualCombined_reads A f r stride s₃ s ((hf₁.trans (hf₂.trans hf₃)).readsEqual hsep)] at hv₄
  refine ⟨s₄,?_,hr₄,hf₁.trans (hf₂.trans (hf₃.trans hf₄)),hv₄⟩
  have hcode : (repaired f r stride M N eps p block hb).surfaceBody.take 15=
      stagePrefix f r stride M N eps p block hb ++ firstStream f r stride M N eps p block hb ++
      secondInput f r stride M N eps p block hb ++ residual f r stride M N eps p block hb := rfl
  rw [hcode]
  simp only [run_append,h₁,h₂,h₃,h₄,Option.bind_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
