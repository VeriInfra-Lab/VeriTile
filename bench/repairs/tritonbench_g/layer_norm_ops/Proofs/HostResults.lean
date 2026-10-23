import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Host
namespace VeriTile.Bench.Repairs.LayerNormOps.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

theorem finish_dw {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (res : Bool) (s : State α) (i : Fin p.columns) (hne : dw≠db) :
    (backwardFinish ops p programs pw pb dw db res s).state.mem dw i.val =
      .mk .fp32 (LayerNormLiger.Host.sumValue ops (partialMatrix p p.dw programs) pw s i.val) := by
  unfold backwardFinish
  cases hb : p.hasBias
  · simp only [hb,Bool.false_eq_true,if_false]
    exact LayerNormLiger.Host.axisSum_read ops (partialMatrix p p.dw programs) pw dw s i
  · simp only [hb,if_true]
    rw [LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hne]
    exact LayerNormLiger.Host.axisSum_read ops (partialMatrix p p.dw programs) pw dw s i

theorem finish_db {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (res : Bool) (s : State α) (i : Fin p.columns)
    (hb : p.hasBias=Bool.true) (hfresh : p.db≠dw) :
    (backwardFinish ops p programs pw pb dw db res s).state.mem db i.val =
      .mk .fp32 (LayerNormLiger.Host.sumValue ops (partialMatrix p p.db programs) pb s i.val) := by
  unfold backwardFinish
  simp only [hb,if_true]
  rw [LayerNormLiger.Host.axisSum_read]
  congr 1
  unfold LayerNormLiger.Host.sumValue
  congr 2
  funext row
  unfold LayerNormLiger.Host.column
  simp only [partialMatrix]
  rw [LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hfresh]

theorem finish_frame {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (res : Bool) (s : State α) (r : RegionName) (o : Nat) (hw : r≠dw) (hb : r≠db) :
    (backwardFinish ops p programs pw pb dw db res s).state.mem r o = s.mem r o := by
  unfold backwardFinish
  cases ht : p.hasBias
  · simp only [ht,Bool.false_eq_true,if_false]
    exact LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hw
  · simp only [ht,if_true]
    rw [LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hb,
      LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hw]

def ForwardRegionOutside (p : ForwardParameters) (r : RegionName) :=
  r≠p.y ∧ r≠p.residualOut ∧ r≠p.mean ∧ r≠p.rstd

theorem forward_grid_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : ForwardParameters) (s : State α) (rows : Nat) (r : RegionName) (o : Nat)
    (h : ForwardRegionOutside p r) :
    (Grid.reference (Forward.reference M p) s rows).mem r o=s.mem r o := by
  apply LayerNormLiger.Grid.frame
  intro pid _ mem
  apply Forward.reference_frame
  exact ⟨Or.inl h.1,fun _ => Or.inl h.2.1,fun _ => Or.inl h.2.2.1,Or.inl h.2.2.2⟩

theorem backward_grid_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s : State α) (programs : Nat) (r : RegionName) (o : Nat)
    (h : Backward.RegionOutside p r) :
    (Grid.reference (Backward.reference M p) s programs).mem r o=s.mem r o := by
  apply LayerNormLiger.Grid.frame
  intro pid _ mem
  exact Backward.reference_frame M p _ r o h

/-- Caller-live cells outside fresh outputs survive the complete public call,
including both kernel grids and both host reductions. -/
theorem composed_frame {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State α)
    (r : RegionName) (o : Nat) (hf : ForwardRegionOutside fp r)
    (hb : Backward.RegionOutside bp r) (hw : r≠dw) (hdb : r≠db) :
    ∃ result, composed M ops fp bp rows programs pw pb dw db s=some result ∧
      result.2.state.mem r o=s.mem r o := by
  rw [composed_complete]
  refine ⟨_,rfl,?_⟩
  rw [finish_frame _ _ _ _ _ _ _ _ _ _ _ hw hdb,
    backward_grid_frame _ _ _ _ _ _ hb]
  exact forward_grid_frame M fp s rows r o hf

#axiomsClean finish_dw
#axiomsClean finish_db
#axiomsClean composed_frame
end
end VeriTile.Bench.Repairs.LayerNormOps.Host
