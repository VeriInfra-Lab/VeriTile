import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyLoop
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyPrologue

/-! Compose both helper ranges and their actual pointer reset. Allocation
windows are restricted to the visited tiles in each stage, including stage 2's
repeated diagonal. The second helper uses preserved caller pointers. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyMiddle
open VeriTile.Triton RealLoads RealLoopFormula
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 1000000

theorem first_bounds (bounds : RegionBounds) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (diagonal : Bool) (h : RealAccessSafety.Windows c Out BM BN BD (if diagonal then 2 else 3) bounds) :
    RealSafetyLoop.Bounds bounds c BN BD (firstLower c.startM BM diagonal) (firstUpper c.startM BM diagonal) := by
  apply RealSafetyLoop.bounds_of_visits bounds c Out BM BN BD (if diagonal then 2 else 3)
    _ _ diagonal h
  cases diagonal <;> simp [firstLower, firstUpper, RealFormula.allVisits, RealFormula.firstVisits]

theorem second_bounds (bounds : RegionBounds) (c : Config) (Out : RegionName) (BM BN BD stage : Nat)
    (h : RealAccessSafety.Windows c Out BM BN BD stage bounds) :
    RealSafetyLoop.Bounds bounds c BN BD (c.startM*BM) ((c.startM+1)*BM) := by
  apply RealSafetyLoop.bounds_of_visits bounds c Out BM BN BD stage _ _ Bool.true h
  simp [RealFormula.allVisits, RealFormula.secondVisits]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyMiddle
