import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoadSafety
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStoreSafety
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealIteration

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStepSafety
open VeriTile.Triton RealLoopState
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (xs : RealMath.Inputs) (pid i : Nat) (hM : 0<p.M) (hN : 0<p.N)
    (hb : RealAccess.Bounds p bounds) (hinput : RealInput.Represents p xs s)
    (h : RealIteration.Ready p xs pid i s) :
    Stmt.TraceSafeList bounds ((RealProgram.body p).drop 2) s := by
  let r := row p pid i
  let c := column p pid i
  let ki := phase p i
  let l := RealLoads.result p s (RealStore.rows p r) (RealStore.columns p c) ki
  have hload := RealLoads.executes p s (RealStore.rows p r) (RealStore.columns p c) ki
    h.ki h.coords.a h.coords.b h.fixed.lanes
  have hdot : stepStmts (RealProgram.dot p) l = some (RealStep.middle p s xs r c ki) :=
    RealDot.executes p l xs (RealStore.rows p r) (RealStore.columns p c) ki
      (RealLoads.a_readback p s xs _ _ ki hinput hM)
      (RealLoads.b_readback p s xs _ _ ki hinput hN)
      (by simp [l,RealLoads.result]; exact h.acc)
  rw [RealStep.matches_source,RealStep.code,List.append_assoc]
  apply TraceProofs.append_safe _ _ (RealLoadSafety.safe p bounds s _ _ ki hM hN hb
    h.ki h.coords.a h.coords.b h.fixed.lanes)
  intro t ht
  rw [hload] at ht
  cases ht
  apply TraceProofs.append_safe _ _ (RealPureSafety.dot_safe p bounds l)
  intro u hu
  rw [hdot] at hu
  cases hu
  exact RealStoreSafety.finish_safe p bounds (RealStep.middle p s xs r c ki)
    r c ki (count p) (RealStep.values p xs r c ki) hb
    (by simp [RealStep.middle,RealLoads.result]; exact h.ki)
    (by simp [RealStep.middle,RealLoads.result,h.fixed.count])
    (by simp [RealStep.middle,RealLoads.result]; exact h.coords.row)
    (by simp [RealStep.middle,RealLoads.result]; exact h.coords.column)
    (by simp [RealStep.middle,RealDot.accumulator,RealStep.values])

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStepSafety
