import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealPureSafety
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealStoreStages
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFinish

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStoreSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (r c : Nat) (values : Fin p.BM → Fin p.BN → ℝ) (hb : RealAccess.Bounds p bounds)
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (r : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (c : Int)))
    (hv : s.regs .real [p.BM,p.BN] "accumulator" = some (Tile.mat fun i j => some (values i j))) :
    Stmt.TraceSafeList bounds (RealStore.code p) s := by
  rw [RealStoreStages.partition,List.append_assoc]
  apply TraceProofs.append_safe _ _ (RealPureSafety.store_prepare_safe p bounds s)
  intro t ht
  rw [RealStoreStages.prepares p s r c values hr hc hv] at ht
  cases ht
  simp only [List.singleton_append]
  apply Stmt.TraceSafeList.cons_intro (RealAccess.output_safe p bounds s r c values hb)
  intro u _
  exact RealPureSafety.reset_safe p bounds u

theorem finish_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (r c ki count : Nat) (values : Fin p.BM → Fin p.BN → ℝ) (hb : RealAccess.Bounds p bounds)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hn : s.regs .nat [] "k_tiles" = some (Tile.scalar count))
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (r : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (c : Int)))
    (hv : s.regs .real [p.BM,p.BN] "accumulator" = some (Tile.mat fun i j => some (values i j))) :
    Stmt.TraceSafeList bounds (RealProgram.store p) s := by
  rw [RealFinish.matches_source]
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    refine ⟨by simp [RealFinish.condition,Op.SafeAt.eq_def],?_⟩
    rw [RealFinish.evaluates s ki count hk hn]
    by_cases hlast : ki+1=count
    · simp only [Tile.scalar,hlast,decide_true,if_true]
      exact safe p bounds s r c values hb hr hc hv
    · simp only [Tile.scalar,hlast,decide_false,Bool.false_eq_true,if_false]
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStoreSafety
