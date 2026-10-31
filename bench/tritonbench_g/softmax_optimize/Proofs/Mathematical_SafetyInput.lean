import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyPure
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyAccess
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SignedPrefix

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport

theorem full_load_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [B] "input_ptrs" = some (pointers X N B i p))
    (hbound : ∀ lane : Fin B, p * N + (i + lane.val) < bounds X) :
    (Op.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none).SafeAt bounds s := by
  apply pointer_load_safe bounds B "input_ptrs" s (pointers X N B i p) hptr
  intro lane
  exact hbound lane.1

theorem tail_load_safe (bounds : RegionBounds) (X : RegionName) (N B i p : Nat)
    (s : BlockState)
    (hptr : s.regs .ptr [B] "input_ptrs" = some (pointers X N B i p))
    (hmask : s.regs .bool [B] "mask" =
      some (Tile.vec fun lane ↦ decide (i + lane.val < N)))
    (hbound : ∀ lane : Fin B, i + lane.val < N → p * N + (i + lane.val) < bounds X) :
    (Op.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
      (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))).SafeAt bounds s := by
  apply masked_pointer_load_safe bounds B "input_ptrs" "mask" s
    (pointers X N B i p) _ hptr hmask
  intro lane hactive
  exact hbound lane.1 (of_decide_eq_true hactive)

/-- The source maximum and the remaining arithmetic have no memory accesses. -/
private theorem source_update_safe (bounds : RegionBounds) (X : RegionName) (N B : Nat)
    (s : BlockState) :
    Stmt.TraceSafeList bounds (sourceMaximum B :: (loopAbody X N B).drop 5) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [loopAbody, List.drop, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl <;>
    simp [sourceMaximum, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
