import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedInputStores
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedInputPreserves
import bench.tritonbench_g.layer_norm_fwd.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def InputPost {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (p : ℝ) (block row : Nat) (s t : State α) : Prop :=
  (f.residualOut=Bool.true → ∀ j : TileIndex [block], j.1.val<N →
    t.mem r.residualOut (row*stride.residualOut+j.1.val)=Cell.mk .real
      (cast A (combinedValues A cmp f s r stride M N p row block j))) ∧
  (f.dropout=Bool.true → f.dropoutMask=Bool.true → ∀ j : TileIndex [block], j.1.val<N →
    t.mem r.dropoutMask (row*N+j.1.val)=Cell.mk .bool
      (keepValue A cmp ((s.mem r.seeds row).read .int) p j.1.val)) ∧
  (f.dropout=Bool.true → f.dropoutMask=Bool.true → f.x1=Bool.true →
    ∀ j : TileIndex [block], j.1.val<N →
    t.mem r.dropoutMask ((M+row)*N+j.1.val)=Cell.mk .bool
      (keepValue A cmp ((s.mem r.seeds (M+row)).read .int) p j.1.val))

theorem input_prefix_post {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (hsep : Domain.ReadWriteSeparated f r) (hrow : s.pids 0<M)
    (hmr : r.dropoutMask ≠ r.residualOut) :
    ∃ t, run (engine A) ((repaired f r stride M N eps p block hb).surfaceBody.take 15) s=some t ∧
      Ready f r stride block (s.pids 0) t ∧ Frame f r stride M N (s.pids 0) s t ∧
      t.regs .real [block] "x"=some (combinedValues A cmp f s r stride M N p (s.pids 0) block) ∧
      InputPost A cmp f r stride M N p block (s.pids 0) s t := by
  obtain ⟨s₁,h₁,hr₁⟩ := prefix_success A f r stride M N eps p block hb s
  obtain ⟨s₂,h₂,hr₂⟩ := firstStream_success A cmp f r stride M N eps p block hb (s.pids 0) s₁ hr₁
  obtain ⟨s₃,h₃,hr₃⟩ := secondInput_success A cmp f r stride M N eps p block hb (s.pids 0) s₂ hr₂
  obtain ⟨s₄,h₄,hr₄⟩ := residual_success A f r stride M N eps p block hb (s.pids 0) s₃ hr₃
  have hrun : run (engine A) ((repaired f r stride M N eps p block hb).surfaceBody.take 15) s=some s₄ := by
    have hcode : (repaired f r stride M N eps p block hb).surfaceBody.take 15=
        stagePrefix f r stride M N eps p block hb ++ firstStream f r stride M N eps p block hb ++
        secondInput f r stride M N eps p block hb ++ residual f r stride M N eps p block hb := rfl
    rw [hcode]
    simp only [run_append,h₁,h₂,h₃,h₄,Option.bind_some]
  obtain ⟨u,hu,hru,hfu,hvu⟩ := input_prefix_values A cmp f r stride M N eps p block hb s hsep
  have he : u=s₄ := Option.some.inj (hu.symm.trans hrun)
  subst u
  have hp := prefix_frame A f r stride M N eps p block hb s s₁ h₁
  have hf₁ : Frame f r stride M N (s.pids 0) s s₁ := by
    intro region offset _; rw [hp]
  have hf₂ := firstStream_frame A cmp f r stride M N eps p block hb (s.pids 0) s₁ s₂ hr₁ h₂
  have hread₂ := (hf₁.trans hf₂).readsEqual hsep
  refine ⟨s₄,hrun,hr₄,hfu,hvu,?_,?_,?_⟩
  · intro hout j hj
    obtain ⟨v₃,hv₃⟩ := hr₃.2.2.2.2.2.2.2
    have hv₄ := residual_value A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ v₃ hv₃ h₄
    have heq := Option.some.inj (hv₄.symm.trans hvu)
    have hh := residual_readback A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ v₃ hv₃ hout h₄ j hj
    change s₄.mem r.residualOut _=Cell.mk .real (cast A ((fun k =>
      if f.residual then op A .add (v₃ k) (raw A s₃ r.residual ((s.pids 0)*stride.residual) N k.1.val)
      else v₃ k) j)) at hh
    rw [heq] at hh
    exact hh
  · intro hd hm j hj
    rw [residual_preserves A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ h₄ _ _ hmr]
    rw [secondInput_preserves A cmp f r stride M N eps p block hb (s.pids 0) s₂ s₃ hr₂ h₃ _ _ (by
      intro _ _ _ h
      exact WholeGrid.mask_halves_disjoint hrow (Contract.inRow_of_col hj) h.2)]
    rw [firstStream_mask_readback A cmp f r stride M N eps p block hb (s.pids 0) s₁ s₂ hr₁ hd hm h₂ j hj]
    rw [hp]
  · intro hd hm hx1 j hj
    rw [residual_preserves A f r stride M N eps p block hb (s.pids 0) s₃ s₄ hr₃ h₄ _ _ hmr]
    rw [secondInput_mask_readback A cmp f r stride M N eps p block hb (s.pids 0) s₂ s₃ hr₂ hd hm hx1 h₃ j hj]
    rw [hread₂ r.seeds (by simp [Domain.readRegion,hd])]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
