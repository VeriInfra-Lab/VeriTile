import bench.tritonbench_g.layer_norm_welfold.Proofs.Grid

namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

structure Configuration where
  block : Nat
  tree : ReductionTree block
  allowed : block = 1024 ∨ block = 2048

theorem Configuration.positive (c : Configuration) : 0 < c.block := by
  rcases c.allowed with h | h <;> omega

def gridMiss (Y Mean Rstd : RegionName) (S D : Nat) (q : RegionName) (off : Nat) : Prop :=
  ∀ row, row < S → rowMiss Y Mean Rstd D row q off

def runTrials {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) : List Configuration → State α → Option (State α)
  | [], s => some s
  | c :: rest, s => do
      let t ← runRows (withEquality M eq32) (rowKernel X W Bias Y Mean Rstd S D c.block c.tree) (List.range S) s
      runTrials M eq32 X W Bias Y Mean Rstd S D rest t

theorem trials_frame {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D : Nat) (sep : Separation X W Bias Y Mean Rstd) (trials : List Configuration) :
    ∃ t, runTrials M eq32 X W Bias Y Mean Rstd S D trials s = some t ∧
      ∀ q off, gridMiss Y Mean Rstd S D q off → t.mem q off = s.mem q off := by
  induction trials generalizing s with
  | nil => exact ⟨s, rfl, fun _ _ _ => rfl⟩
  | cons c cs ih =>
    obtain ⟨u, hu, he⟩ := whole_grid M eq32 s X W Bias Y Mean Rstd S D c.block c.tree c.positive sep (List.range S)
    obtain ⟨t, ht, hf⟩ := ih u
    refine ⟨t, by simp [runTrials, hu, ht], ?_⟩
    intro q off hm
    rw [hf q off hm]
    exact he.frame q off (fun row hr => hm row (List.mem_range.mp hr))

/-- Trial runs cannot alter the logical input of the selected final run. No
bitwise equality between different configurations is assumed. -/
theorem selected_after_trials {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D : Nat) (sep : Separation X W Bias Y Mean Rstd) (trials : List Configuration) (last : Configuration) :
    ∃ u t, runTrials M eq32 X W Bias Y Mean Rstd S D trials s = some u ∧
      runRows (withEquality M eq32) (rowKernel X W Bias Y Mean Rstd S D last.block last.tree) (List.range S) u = some t ∧
      GridEffect M s t X W Bias Y Mean Rstd D last.block last.tree (List.range S) := by
  obtain ⟨u, hu, hf⟩ := trials_frame M eq32 s X W Bias Y Mean Rstd S D sep trials
  obtain ⟨t, ht, he⟩ := whole_grid M eq32 u X W Bias Y Mean Rstd S D last.block last.tree last.positive sep (List.range S)
  have hinput : ∀ q, q = X ∨ q = W ∨ q = Bias → u.mem q = s.mem q := by
    intro q hq
    funext off
    apply hf q off
    intro row hr
    exact ⟨.inl (sep.inputs q hq).1, .inl (sep.inputs q hq).2.1, .inl (sep.inputs q hq).2.2⟩
  have hm : ∀ row, meanValue M (atRow u row) X D last.block last.tree =
      meanValue M (atRow s row) X D last.block last.tree :=
    fun row => mean_congr M u s X D last.block row last.tree (hinput X (.inl rfl))
  have hv : ∀ row, inverseValue M (atRow u row) X D last.block last.tree =
      inverseValue M (atRow s row) X D last.block last.tree :=
    fun row => inverse_congr M u s X D last.block row last.tree (hinput X (.inl rfl))
  refine ⟨u, t, hu, ht, ?_, ?_, ?_, ?_⟩
  · intro row hr; simpa only [hm] using he.mean row hr
  · intro row hr; simpa only [hv] using he.rstd row hr
  · intro row hr col hc
    simpa only [rowOutput, hm, hv, normalizeValue, hinput X (.inl rfl),
      hinput W (.inr (.inl rfl)), hinput Bias (.inr (.inr rfl))] using he.output row hr col hc
  · intro q off hm
    rw [he.frame q off hm]
    exact hf q off (fun row hr => hm row (List.mem_range.mpr hr))

#axiomsClean selected_after_trials
end
end VeriTile.Bench.Repairs.LayerNormWelfold
