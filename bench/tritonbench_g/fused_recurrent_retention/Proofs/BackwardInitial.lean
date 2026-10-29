import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardHeader
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstLoop

/-! Actual backward source initialization establishes the first-loop invariant
for both seed branches. No prepared carry or caller register assumptions. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardInitial
open VeriTile.Triton Primitives BackwardFirstState BackwardHeaderPrimitives BackwardHeaderSource
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def state (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) : BlockState :=
  let h := BackwardHeader.state p r initial
  if hasSeed then
    ((h.setReg "mask_kv" .bool [p.BK,p.BV] (stateMask p (launch initial))).setReg
      "p_init_s" .ptr [p.BK,p.BV] (statePointers p (launch initial) r.initial)).setReg
      "h" .real [p.BK,p.BV] (carryTile p (launch initial) initial r Bool.true 0)
  else h

theorem seed_body (p : Layout) (r : BackwardRegions) (initial : BlockState) :
    stepStmts
      [.assign .bool [p.BK,p.BV] "mask_kv" (maskKV p),
       .assign .ptr [p.BK,p.BV] "p_init_s" (statePointerOp p r.initial),
       .assign .real [p.BK,p.BV] "h" (seedAdd p)] (BackwardHeader.state p r initial) =
      some (state p r Bool.true initial) := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (state_mask p (launch initial) _ (by simp [BackwardHeader.state]) (by simp [BackwardHeader.state])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (state_pointer p (launch initial) r.initial _ (by simp [BackwardHeader.state])
      (by simp [BackwardHeader.state]) (by simp [BackwardHeader.state])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (seed_add p (launch initial) r initial _ (by simp) (by simp)
      (by simp [BackwardHeader.state]) (by intro off; rfl)))]
  rw [stepStmts.nil]
  rfl

theorem seed_executes (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) :
    stepStmts (seedCode p r hasSeed) (BackwardHeader.state p r initial) =
      some (state p r hasSeed initial) := by
  have hif : stepStmt (.ifThen (.constBool hasSeed)
      [.assign .bool [p.BK,p.BV] "mask_kv" (maskKV p),
       .assign .ptr [p.BK,p.BV] "p_init_s" (statePointerOp p r.initial),
       .assign .real [p.BK,p.BV] "h" (seedAdd p)]) (BackwardHeader.state p r initial) =
      some (state p r hasSeed initial) := by
    have hc (b : Bool) (v : BlockState) : evalOp (.constBool b) v = some (Tile.scalar b) := by
      simp only [evalOp.eq_def]
    cases hasSeed
    · simp only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,Bool.false_eq_true,↓reduceIte,state]
    · simpa only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,↓reduceIte] using seed_body p r initial
  rw [seedCode,stepStmts.cons_some hif,stepStmts.nil]

theorem executes (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) (initial : BlockState) :
    stepStmts (BackwardSource.initialCode p r hasSeed swapped) initial =
      some (state p r hasSeed initial) := by
  rw [BackwardHeaderSource.initial_exact,
    stepStmts.append_some (BackwardHeader.executes p r swapped initial),seed_executes]

theorem ready (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) :
    Ready p (launch initial) initial r hasSeed 0 (state p r hasSeed initial) := by
  cases hasSeed
  · exact BackwardHeader.ready p r initial
  · constructor <;> simp [state,BackwardHeader.state]

theorem memory (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) :
    (state p r hasSeed initial).mem = initial.mem := by cases hasSeed <;> rfl

theorem invariant (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial : BlockState) :
    BackwardFirstLoop.Holds p (launch initial) initial r hasSeed 0 (state p r hasSeed initial) := by
  refine ⟨ready p r hasSeed initial,?_,?_,?_,?_,?_,?_⟩
  · intro off; simp only [BlockState.readMem,memory]
  · intro off; simp only [BlockState.readMem,memory]
  · intro off; simp only [BlockState.readMem,memory]
  · intro off; simp only [BlockState.readMem,memory]
  · intro t k ht; omega
  · intro region off _; rw [memory]

#axiomsClean seed_body
#axiomsClean seed_executes
#axiomsClean executes
#axiomsClean ready
#axiomsClean memory
#axiomsClean invariant
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardInitial
