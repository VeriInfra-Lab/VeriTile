import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Allocation
namespace VeriTile.Bench.Repairs.LayerNormOps.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

theorem finish_active_frame {α : Type} [Inhabited α] (ops : Scalar α) (p : BackwardParameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName)
    (res : Bool) (s : State α) (r : RegionName) (o : Nat) (hw : r≠dw)
    (hb : p.hasBias=Bool.true → r≠db) :
    (backwardFinish ops p programs pw pb dw db res s).state.mem r o = s.mem r o := by
  unfold backwardFinish
  cases ht : p.hasBias
  · simp only [ht,Bool.false_eq_true,if_false]
    exact LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hw
  · simp only [ht,if_true]
    rw [LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ (hb ht),
      LayerNormLiger.Host.axisSum_frame _ _ _ _ _ _ _ hw]

theorem forward_active_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : ForwardParameters) (s : State α) (rows : Nat) (r : RegionName) (o : Nat)
    (h : r≠p.y ∧ (p.storeResidual=Bool.true → r≠p.residualOut) ∧
      (p.rms=Bool.false → r≠p.mean) ∧ r≠p.rstd) :
    (Grid.reference (Forward.reference M p) s rows).mem r o=s.mem r o := by
  apply LayerNormLiger.Grid.frame
  intro pid _ mem
  apply Forward.reference_frame
  exact ⟨Or.inl h.1,fun hs => Or.inl (h.2.1 hs),fun hs => Or.inl (h.2.2.1 hs),Or.inl h.2.2.2⟩

/-- Final caller-visible frame from the ordinary allocator boundary; inactive
optional outputs introduce no artificial freshness requirement. -/
theorem caller_frame {α : Type} [Inhabited α] (M : Algebra α) (ops : Scalar α)
    (fp : ForwardParameters) (bp : BackwardParameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (s : State α)
    (live : List RegionName) (ha : Scope.Allocations fp bp dw db live)
    (r : RegionName) (o : Nat) (hr : r∈live) :
    ∃ result, composed M ops fp bp rows programs pw pb dw db s=some result ∧
      result.2.state.mem r o=s.mem r o := by
  obtain ⟨hf,hb,hw,hdb⟩ := Scope.caller_outside fp bp dw db live ha r hr
  rw [composed_complete]
  refine ⟨_,rfl,?_⟩
  rw [finish_active_frame _ _ _ _ _ _ _ _ _ _ _ hw hdb,
    backward_grid_frame _ _ _ _ _ _ hb]
  exact forward_active_frame M fp s rows r o hf

#axiomsClean caller_frame
end
end VeriTile.Bench.Repairs.LayerNormOps.Host
