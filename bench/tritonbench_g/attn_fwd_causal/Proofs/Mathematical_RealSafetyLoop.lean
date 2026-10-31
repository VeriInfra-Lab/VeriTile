import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyParts

/-! Safety follows the actual helper range and its pointer invariant.
Every visited K/V lane is bounded only when its source mask is active;
each visited key scale is bounded unconditionally. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyLoop
open VeriTile.Triton RealLoads RealMemory RealLoop
open AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000

structure Bounds (bounds : RegionBounds) (c : Config) (BN BD lower upper : Nat) : Prop where
  keys : ∀ i, i < RealFormula.tripCount lower upper BN → ∀ (j : Fin BN) (d : Fin BD),
    cursor lower BN i+j.val < c.context → d.val < c.activeHead →
    c.base+d.val+j.val*c.strideKN+c.headDim*cursor lower BN i < bounds c.K
  scales : ∀ i, i < RealFormula.tripCount lower upper BN →
    c.keyScaleBase+cursor lower BN i/BN < bounds c.KS
  values : ∀ i, i < RealFormula.tripCount lower upper BN → ∀ (j : Fin BN) (d : Fin BD),
    cursor lower BN i+j.val < c.context → d.val < c.activeHead →
    c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*cursor lower BN i < bounds c.V

theorem visit_mem (lower upper BN i : Nat) (diagonal : Bool)
    (hi : i < RealFormula.tripCount lower upper BN) :
    (⟨cursor lower BN i,diagonal⟩ : RealFormula.Visit) ∈ RealFormula.visits lower upper BN diagonal := by
  exact List.mem_map.mpr ⟨i, List.mem_range.mpr hi, rfl⟩

theorem bounds_of_visits (bounds : RegionBounds) (c : Config) (Out : RegionName)
    (BM BN BD stage lower upper : Nat) (diagonal : Bool)
    (h : RealAccessSafety.Windows c Out BM BN BD stage bounds)
    (hsub : RealFormula.visits lower upper BN diagonal ⊆ RealFormula.allVisits c.startM BM BN stage) :
    Bounds bounds c BN BD lower upper := by
  exact ⟨fun i hi => h.key _ (hsub (visit_mem lower upper BN i diagonal hi)),
    fun i hi => h.keyScale _ (hsub (visit_mem lower upper BN i diagonal hi)),
    fun i hi => h.value _ (hsub (visit_mem lower upper BN i diagonal hi))⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyLoop
