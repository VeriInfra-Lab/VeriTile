import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeNormalizeLoop

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Inputs may alias one another. Each allocated output is fresh. -/
structure Separation (X W Bias Y Mean Rstd : RegionName) : Prop where
  mean_rstd : Mean ≠ Rstd
  mean_y : Mean ≠ Y
  rstd_y : Rstd ≠ Y
  inputs : ∀ q, q = X ∨ q = W ∨ q = Bias → q ≠ Mean ∧ q ≠ Rstd ∧ q ≠ Y

def rowValue {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X : RegionName) (D B : Nat) (tree : ReductionTree B) : WelfordTriple α :=
  treeValue M eq32 (allLaneValues M s X D B) tree

def rowKernel (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  .mk [X, W, Bias] [Y, Mean, Rstd]
    ((statisticsKernel X Mean Rstd S D B tree).surfaceBody ++
      normalizeCode X W Bias Y D B 0 (blockCount D B))

def rowMiss (Y Mean Rstd : RegionName) (D row : Nat) (q : RegionName) (off : Nat) : Prop :=
  (q ≠ Mean ∨ off ≠ row) ∧ (q ≠ Rstd ∨ off ≠ row) ∧
    (q ≠ Y ∨ ∀ col, col < D → off ≠ col + D * row)

structure RowEffect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) : Prop where
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  mean : t.mem Mean (s.pids 0) = .mk .real (rowValue M eq32 s X D B tree).mean
  rstd : t.mem Rstd (s.pids 0) = .mk .real (rstdValue M D (rowValue M eq32 s X D B tree))
  output : ∀ col, col < D → t.mem Y (col + D * s.pids 0) =
    .mk .bf16 (normalizeValue M s X W Bias D (s.pids 0) col
      (rowValue M eq32 s X D B tree).mean (rstdValue M D (rowValue M eq32 s X D B tree)))
  frame : ∀ q off, rowMiss Y Mean Rstd D (s.pids 0) q off → t.mem q off = s.mem q off

theorem statistics_frame {α : Type} (M : Algebra α) (s : State α)
    (Mean Rstd : RegionName) (D : Nat) (v : WelfordTriple α) (q : RegionName) (off : Nat)
    (hm : q ≠ Mean ∨ off ≠ s.pids 0) (hr : q ≠ Rstd ∨ off ≠ s.pids 0) :
    (statisticsMemory M s Mean Rstd D v).mem q off = s.mem q off := by
  unfold statisticsMemory
  rw [State.write_other _ _ _ _ _ _ hr, State.write_other _ _ _ _ _ _ hm]

theorem whole_row {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (hB : 0 < B)
    (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final, FP.Structural.exec (withEquality M eq32) (rowKernel X W Bias Y Mean Rstd S D B tree) s = some final ∧
      RowEffect M eq32 s final X W Bias Y Mean Rstd D B tree := by
  obtain ⟨u, hu, hs⟩ := whole_statistics_ready M eq32 s X Mean Rstd S D B tree
  obtain ⟨t, ht, hreg, hout, hframe⟩ := normalize_whole_row M eq32 u X W Bias Y D B (s.pids 0)
    _ _ hB (sep.inputs X (.inl rfl)).2.2 (sep.inputs W (.inr (.inl rfl))).2.2
    (sep.inputs Bias (.inr (.inr rfl))).2.2 hs.mean hs.rstd hs.row
  have hinput : ∀ q, q = X ∨ q = W ∨ q = Bias → u.mem q = s.mem q := by
    intro q hq
    funext off
    rw [hs.mem]
    exact statistics_frame M s Mean Rstd D _ q off (.inl (sep.inputs q hq).1) (.inl (sep.inputs q hq).2.1)
  refine ⟨t, ?_, ?_⟩
  · simp only [FP.Structural.exec, ComputeKernel.surfaceBody] at hu
    simpa only [rowKernel, FP.Structural.exec, ComputeKernel.surfaceBody, run_append, hu, Option.bind, ht]
  · refine ⟨hreg.pids.trans hs.pids, hreg.undef.trans hs.undef, ?_, ?_, ?_, ?_⟩
    · rw [hframe Mean (s.pids 0) (.inl sep.mean_y), hs.mem]
      simp [statisticsMemory, State.write, sep.mean_rstd, rowValue]
    · rw [hframe Rstd (s.pids 0) (.inl sep.rstd_y), hs.mem]
      simp [statisticsMemory, rowValue]
    · intro col hc
      have hv := hout col hc
      simpa only [normalizeValue, hinput X (.inl rfl), hinput W (.inr (.inl rfl)),
        hinput Bias (.inr (.inr rfl)), rowValue] using hv
    · intro q off hm
      rw [hframe q off hm.2.2, hs.mem]
      exact statistics_frame M s Mean Rstd D _ q off hm.1 hm.2.1

#axiomsClean whole_row
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
