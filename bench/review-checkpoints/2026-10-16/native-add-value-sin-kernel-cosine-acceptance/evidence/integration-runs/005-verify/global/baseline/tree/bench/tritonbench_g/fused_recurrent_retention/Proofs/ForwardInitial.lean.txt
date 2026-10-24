import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardHeader
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardLoop

/-! Source initialization establishes the loop invariant from the original
memory snapshot for both seed flags and both candidates. No carry premise. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardInitial
open VeriTile.Triton Primitives MaskedMath ForwardState StatePrimitives ForwardHeaderSource
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def state (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) : BlockState :=
  let h := ForwardHeader.state p r initial
  if hasSeed then
    (h.setReg "p_init_s" .ptr [p.BV,p.BK] (statePointers p (launch initial) r.initial)).setReg
      "h" .real [p.BV,p.BK] (carryTile p (launch initial) initial r Bool.true 0)
  else h

theorem seed_body (p : Layout) (r : ForwardRegions) (initial : BlockState) :
    stepStmts
      [.assign .ptr [p.BV,p.BK] "p_init_s" (statePointerOp p r.initial),
       .assign .real [p.BV,p.BK] "h" (seedAdd p)] (ForwardHeader.state p r initial) =
        some (state p r Bool.true initial) := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.state_pointer p (launch initial) r.initial _
      (by simp [ForwardHeader.state]) (by simp [ForwardHeader.state])
      (by simp [ForwardHeader.state])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (StatePrimitives.seed_add p (launch initial) r initial _ (by simp)
      (by simp [ForwardHeader.state]) (by simp [ForwardHeader.state])
      (by intro off; rfl)))]
  rw [stepStmts.nil]
  rfl

theorem seed_executes (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) :
    stepStmts (seedCode p r hasSeed) (ForwardHeader.state p r initial) =
      some (state p r hasSeed initial) := by
  have hif : stepStmt (.ifThen (.constBool hasSeed)
      [.assign .ptr [p.BV,p.BK] "p_init_s" (statePointerOp p r.initial),
       .assign .real [p.BV,p.BK] "h" (seedAdd p)]) (ForwardHeader.state p r initial) =
        some (state p r hasSeed initial) := by
    have hc (b : Bool) (v : BlockState) :
        evalOp (.constBool b) v = some (Tile.scalar b) := by
      simp only [evalOp.eq_def]
    cases hasSeed
    · simp only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,
        Bool.false_eq_true,↓reduceIte,state]
    · simpa only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,↓reduceIte]
        using seed_body p r initial
  rw [seedCode,stepStmts.cons_some hif,stepStmts.nil]

theorem executes (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool)
    (initial : BlockState) :
    stepStmts (ForwardSource.initialCode p r hasSeed storeFinal swapped) initial =
      some (state p r hasSeed initial) := by
  rw [ForwardHeaderSource.initial_exact,
    stepStmts.append_some (ForwardHeader.executes p r swapped initial),seed_executes]

theorem ready (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) :
    Ready p (launch initial) initial r hasSeed 0 (state p r hasSeed initial) := by
  cases hasSeed
  · exact ForwardHeader.ready p r initial
  · constructor <;> simp [state,ForwardHeader.state,StatePrimitives.stateMask]

theorem memory (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) :
    (state p r hasSeed initial).mem = initial.mem := by
  cases hasSeed <;> rfl

theorem invariant (p : Layout) (r : ForwardRegions) (hasSeed : Bool) (initial : BlockState) :
    ForwardLoop.Holds p (launch initial) initial r hasSeed 0 (state p r hasSeed initial) := by
  refine ⟨ready p r hasSeed initial,?_,?_,?_,?_,?_⟩
  · intro off; simp only [BlockState.readMem,memory]
  · intro off; simp only [BlockState.readMem,memory]
  · intro off; simp only [BlockState.readMem,memory]
  · intro t j ht; omega
  · intro region off _; rw [memory]

#axiomsClean seed_body
#axiomsClean seed_executes
#axiomsClean executes
#axiomsClean ready
#axiomsClean memory
#axiomsClean invariant
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardInitial
