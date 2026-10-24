import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondValues
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Memory

/-! Complete reverse-loop iteration: the two actual masked gradient stores and
all six original pointer subtractions follow the eight numerical assignments.
The final iteration retains its actual truncated pointer values. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondStep
open VeriTile.Triton Primitives MaskedMath BackwardSecondState
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1400000
set_option linter.unusedSimpArgs false

theorem key_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r (t+1) u) :
    stepStmt (.store .real [p.BK] (.ptr (.ref .ptr [p.BK] "p_dk"))
      (.ref .real [p.BK] "d_k") (.mask (.ref .bool [p.BK] "mask_bk")))
      (beforeStores p l initial r t u) =
        some (scatterK p l initial r t (beforeStores p l initial r t u)) := by
  exact Memory.store_vector p.BK r.dk "p_dk" "d_k" "mask_bk"
    (fun k => keyGradientAddress p l t k.val) (MaskedMath.dk p l initial r t)
    (fun k => activeKey p l k.val) _
    (by simpa [beforeStores,keyGradientPointers] using hu.dk (by omega))
    (by simp [beforeStores]) (by simp [beforeStores,hu.maskK])

theorem value_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r (t+1) u) :
    stepStmt (.store .real [p.BV] (.ptr (.ref .ptr [p.BV] "p_dv"))
      (.ref .real [p.BV] "d_v") (.mask (.ref .bool [p.BV] "mask_bv")))
      (scatterK p l initial r t (beforeStores p l initial r t u)) =
        some (scatterV p l initial r t
          (scatterK p l initial r t (beforeStores p l initial r t u))) := by
  exact Memory.store_vector p.BV r.dv "p_dv" "d_v" "mask_bv"
    (fun j => outputAddress p l t j.val) (MaskedMath.dv p l initial r t)
    (fun j => activeValue p l j.val) _
    (by simpa [scatterK,beforeStores,outputPointers] using hu.dv (by omega))
    (by simp [scatterK,beforeStores]) (by simp [scatterK,beforeStores,hu.maskV])

theorem executes (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (t : Nat) (ht : t < p.T) (u : BlockState)
    (hu : Ready p l initial r (t+1) u)
    (hqmem : ∀ off, u.readMem r.q off = initial.readMem r.q off)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off)
    (hdmem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off) :
    stepStmts (BackwardSource.secondBody p) u = some (result p l initial r t u) := by
  rw [←List.take_append_drop 8 (BackwardSource.secondBody p),
    stepStmts.append_some (BackwardSecondValues.before_stores p l initial r t ht u hu
      hqmem hkmem hvmem hdmem)]
  dsimp only [BackwardSource.secondBody,List.drop]
  rw [stepStmts.cons_some (key_store p l initial r t u hu)]
  rw [stepStmts.cons_some (value_store p l initial r t u hu)]
  dsimp only [BackwardSource.retreat]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BV r.do_ "p_do" (fun j => valueAddress p l t j.val) p.DV _
    (by simpa [scatterK,scatterV,beforeStores,valuePointers] using hu.dout (by omega))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BK r.q "p_q" (fun k => keyAddress p l t k.val) p.DK _
    (by simpa [scatterK,scatterV,beforeStores,keyPointers] using hu.q (by omega))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BK r.k "p_k" (fun k => keyAddress p l t k.val) p.DK _
    (by simpa [scatterK,scatterV,beforeStores,keyPointers] using hu.k (by omega))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BV r.v "p_v" (fun j => valueAddress p l t j.val) p.DV _
    (by simpa [scatterK,scatterV,beforeStores,valuePointers] using hu.v (by omega))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BK r.dk "p_dk" (fun k => keyGradientAddress p l t k.val) p.DK _
    (by simpa [scatterK,scatterV,beforeStores,keyGradientPointers] using hu.dk (by omega))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (BackwardSecondPrimitives.retreat_pointer
    p.BV r.dv "p_dv" (fun j => outputAddress p l t j.val) p.DV _
    (by simpa [scatterK,scatterV,beforeStores,outputPointers] using hu.dv (by omega))))]
  rw [stepStmts.nil]
  rfl

#axiomsClean key_store
#axiomsClean value_store
#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondStep
