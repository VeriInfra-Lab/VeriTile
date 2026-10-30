import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Statistics
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem recurrence_equality {α : Type} [Inhabited α] (M : Algebra α) (eq : α→α→Bool)
    (phase : Bool) (s : State α) (X : RegionName) (D B : Nat) (mean : α)
    (start count : Nat) (previous : Fin B→α) :
    recurrence (withEquality M eq) phase s X D B mean start count previous =
      recurrence M phase s X D B mean start count previous := by
  induction count generalizing start previous with
  | zero => rfl
  | succ count ih => simpa only [recurrence] using ih (start+1) (advance M phase s X D B start mean previous)

theorem tree_equality {α : Type} {B : Nat} (M : Algebra α) (eq : α→α→Bool)
    (values : Fin B→α) (tree : ReductionTree B) :
    treeValue (withEquality M eq) values tree = treeValue M values tree := by
  induction tree with
  | leaf i => rfl
  | node l r il ir =>
    simp only [treeValue,il,ir]
    rfl

theorem mean_equality {α : Type} [Inhabited α] (M : Algebra α) (eq : α→α→Bool)
    (s : State α) (X : RegionName) (D B : Nat) (tree : ReductionTree B) :
    meanValue (withEquality M eq) s X D B tree = meanValue M s X D B tree := by
  unfold meanValue allLanes
  rw [recurrence_equality,tree_equality]
  rfl

theorem inverse_equality {α : Type} [Inhabited α] (M : Algebra α) (eq : α→α→Bool)
    (s : State α) (X : RegionName) (D B : Nat) (tree : ReductionTree B) :
    inverseValue (withEquality M eq) s X D B tree = inverseValue M s X D B tree := by
  unfold inverseValue
  rw [mean_equality]
  unfold allLanes
  rw [recurrence_equality,tree_equality]
  rfl

structure RowEffect {α : Type} [Inhabited α] (M : Algebra α)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) : Prop where
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.mem Mean (s.pids 0)=.mk .real (meanValue M s X D B tree)
  rstd : t.mem Rstd (s.pids 0)=.mk .real (inverseValue M s X D B tree)
  output : ∀ col, col<D → t.mem Y (col+D*s.pids 0)=.mk .bf16
    (normalizeValue M s X W Bias D (s.pids 0) col (meanValue M s X D B tree) (inverseValue M s X D B tree))
  frame : ∀ q off, rowMiss Y Mean Rstd D (s.pids 0) q off → t.mem q off=s.mem q off

theorem statistics_frame {α : Type} (s : State α) (Mean Rstd : RegionName)
    (mean inverse : α) (q : RegionName) (off : Nat)
    (hm : q≠Mean ∨ off≠s.pids 0) (hr : q≠Rstd ∨ off≠s.pids 0) :
    ((s.write Mean (s.pids 0) (.mk .real mean)).write Rstd (s.pids 0) (.mk .real inverse)).mem q off=s.mem q off := by
  rw [State.write_other _ _ _ _ _ _ hr,State.write_other _ _ _ _ _ _ hm]

theorem whole_row {α : Type} [Inhabited α] (M : Algebra α) (eq : α→α→Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) (hB : 0<B) (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final, FP.Structural.exec (withEquality M eq) (rowKernel X W Bias Y Mean Rstd S D B tree) s=some final ∧
      RowEffect M s final X W Bias Y Mean Rstd D B tree := by
  obtain ⟨u,hu,hs⟩ := whole_statistics (withEquality M eq) s X Mean Rstd S D B tree (sep.inputs X (.inl rfl)).1
  rw [mean_equality,inverse_equality] at hs
  obtain ⟨t,ht,hreg,hout,hframe⟩ := normalize_whole_row M eq u X W Bias Y D B (s.pids 0)
    _ _ hB (sep.inputs X (.inl rfl)).2.2 (sep.inputs W (.inr (.inl rfl))).2.2
    (sep.inputs Bias (.inr (.inr rfl))).2.2 hs.mean hs.inverse hs.row
  have hinput : ∀ q, q=X ∨ q=W ∨ q=Bias → u.mem q=s.mem q := by
    intro q hq
    funext off
    rw [hs.mem]
    exact statistics_frame s Mean Rstd _ _ q off (.inl (sep.inputs q hq).1) (.inl (sep.inputs q hq).2.1)
  refine ⟨t,?_,?_⟩
  · simp only [FP.Structural.exec,ComputeKernel.surfaceBody] at hu
    simpa only [rowKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run_append,hu,Option.bind,ht]
  · refine ⟨hreg.pids.trans hs.pids,hreg.undef.trans hs.undef,?_,?_,?_,?_⟩
    · rw [hframe Mean (s.pids 0) (.inl sep.mean_y),hs.mem]
      simp [State.write,sep.mean_rstd]
    · rw [hframe Rstd (s.pids 0) (.inl sep.rstd_y),hs.mem]
      simp [State.write]
    · intro col hc
      simpa only [normalizeValue,hinput X (.inl rfl),hinput W (.inr (.inl rfl)),
        hinput Bias (.inr (.inr rfl))] using hout col hc
    · intro q off hm
      rw [hframe q off hm.2.2,hs.mem]
      exact statistics_frame s Mean Rstd _ _ q off hm.1 hm.2.1

#axiomsClean whole_row
end
end VeriTile.Bench.Repairs.LayerNormWelfold
