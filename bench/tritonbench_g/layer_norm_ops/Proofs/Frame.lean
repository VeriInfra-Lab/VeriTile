import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardModel
import bench.tritonbench_g.layer_norm_liger.Proofs.FoldFacts
import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
namespace VeriTile.Bench.Repairs.LayerNormOps
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
namespace Forward

theorem scatter_frame {α : Type} (p : ForwardParameters) (row : Nat)
    (out : RegionName) (stride : Nat) (values : TileIndex [p.block] → Cell α)
    (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ out ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o ≠ row*stride+i.1.val) :
    (scatter p row out stride values s).mem r o = s.mem r o := by
  unfold scatter
  apply State.masked_scatter_frame
  exact h.elim Or.inl (fun hh => Or.inr (fun i _ hi => hh i hi))

def Outside {α : Type} (p : ForwardParameters) (s : State α) (r : RegionName) (o : Nat) :=
  (r ≠ p.y ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o ≠ s.pids 0*p.yStride+i.1.val) ∧
  (p.storeResidual = Bool.true → r ≠ p.residualOut ∨ ∀ i : TileIndex [p.block],
    i.1.val < p.columns → o ≠ s.pids 0*p.residualOutStride+i.1.val) ∧
  (p.rms = Bool.false → r ≠ p.mean ∨ o ≠ s.pids 0) ∧
  (r ≠ p.rstd ∨ o ≠ s.pids 0)

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : ForwardParameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    (reference M p s).mem r o = s.mem r o := by
  have hres : (residualState M p s).mem r o = s.mem r o := by
    cases hs : p.storeResidual
    · simp [residualState, hs]
    · simpa only [residualState, hs, if_true] using
        scatter_frame p (s.pids 0) p.residualOut p.residualOutStride
          (fun i => stored M (z M p s i)) s r o (h.2.1 hs)
  have hr : ¬(r=p.rstd ∧ o=s.pids 0) := by have := h.2.2.2; tauto
  unfold reference
  rw [scatter_frame _ _ _ _ _ _ _ _ h.1]
  unfold statisticsState
  cases hm : p.rms
  · have hn : ¬(r=p.mean ∧ o=s.pids 0) := by have := h.2.2.1 hm; tauto
    simpa only [hm, Bool.false_eq_true, if_false, State.write, hr, hn] using hres
  · simpa only [hm, if_true, State.write, hr, if_false] using hres
end Forward
namespace Backward

theorem scatter_frame {α : Type} (p : BackwardParameters) (row : Nat)
    (out : RegionName) (stride : Nat) (values : TileIndex [p.block] → Cell α)
    (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ out ∨ ∀ i : TileIndex [p.block], i.1.val < p.columns → o ≠ row*stride+i.1.val) :
    (scatter p row out stride values s).mem r o = s.mem r o := by
  unfold scatter
  apply State.masked_scatter_frame
  exact h.elim Or.inl (fun hh => Or.inr (fun i _ hi => hh i hi))

def RegionOutside (p : BackwardParameters) (r : RegionName) :=
  r ≠ p.dx ∧ r ≠ p.dw ∧ (p.hasBias = Bool.true → r ≠ p.db) ∧
    (p.recompute = Bool.true → r ≠ p.y) ∧ (p.storeDres = Bool.true → r ≠ p.dresIn)

theorem update_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (row : Nat) (a : Acc α p) (r : RegionName) (o : Nat)
    (h : RegionOutside p r) :
    (update M p row a).state.mem r o = a.state.mem r o := by
  have hy : (recomputed M p a row).mem r o = a.state.mem r o := by
    cases hy : p.recompute
    · simp [recomputed, hy]
    · simpa only [recomputed, hy, if_true] using
        scatter_frame p row p.y p.yStride (fun i => stored M (yValue M p a row i)) a.state
          r o (Or.inl (h.2.2.2.1 hy))
  unfold update
  rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.1)]
  cases hs : p.storeDres
  · simpa only [hs, Bool.false_eq_true, if_false] using hy
  · simp only [hs, if_true]
    rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl (h.2.2.2.2 hs))]
    exact hy

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s : State α) (r : RegionName) (o : Nat)
    (h : RegionOutside p r) : (reference M p s).mem r o = s.mem r o := by
  have hloop : (accumulated M p s).state.mem r o = s.mem r o := by
    apply LayerNormLiger.Folds.preserves (update M p)
      (fun a => a.state.mem r o = s.mem r o)
    · intro row a ha
      rw [update_frame M p row a r o h]
      exact ha
    · rfl
  unfold reference finish
  cases hb : p.hasBias
  · simp only [hb, Bool.false_eq_true, if_false]
    rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.2.1)]
    exact hloop
  · simp only [hb, if_true]
    rw [scatter_frame _ _ _ _ _ _ _ _ (Or.inl (h.2.2.1 hb)),
      scatter_frame _ _ _ _ _ _ _ _ (Or.inl h.2.1)]
    exact hloop
end Backward
#axiomsClean Forward.reference_frame
#axiomsClean Backward.reference_frame
end
end VeriTile.Bench.Repairs.LayerNormOps
