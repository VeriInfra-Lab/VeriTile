import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_SourceBridge

namespace VeriTile.Bench.TritonBenchG.SpinningLock.Frontend

/-- Pure tree transformation used only by compiler rejection regressions. -/
def mapTree (f : Statement → Statement) : Statement → Statement
  | .forRange i a b body => f (.forRange i a b (body.map (mapTree f)))
  | .branch c yes no => f (.branch c (yes.map (mapTree f)) (no.map (mapTree f)))
  | .whileLoop c body => f (.whileLoop c (body.map (mapTree f)))
  | s => f s

def changeCas : Statement → Statement
  | .whileLoop (.casNe p a b _) body => .whileLoop (.casNe p a b (.literal 0)) body
  | s => s

def removeAdd : Statement → Statement
  | .addFP32 _ _ => .pass
  | s => s

def removePoll : Statement → Statement
  | .whileLoop (.casNe ..) _ => .pass
  | s => s

def changeNext : Statement → Statement
  | .integer "next_pid" (.add .pid (.literal 1)) => .integer "next_pid" (.literal 7)
  | s => s

def reversePublication : Statement → Statement
  | .branch c yes [.range r rows, .range n k, .pointer p a, .storeFP32 b v mask, .xchg l x] =>
      .branch c yes [.range r rows, .range n k, .pointer p a, .xchg l x, .storeFP32 b v mask]
  | s => s

theorem changed_cas_condition_rejected :
    compile (originalTypedSource.map (mapTree changeCas)) = none := by
  simp only [originalTypedSource, ↓List.map_cons, ↓List.map_nil, mapTree.eq_def, changeCas]
  decide +kernel

theorem missing_add_rejected :
    compile (originalTypedSource.map (mapTree removeAdd)) = none := by
  simp only [originalTypedSource, ↓List.map_cons, ↓List.map_nil, mapTree.eq_def, removeAdd]
  decide +kernel

theorem ready_lock_replacement_rejected :
    compile (originalTypedSource.map (mapTree removePoll)) = none := by
  simp only [originalTypedSource, ↓List.map_cons, ↓List.map_nil, mapTree.eq_def, removePoll]
  decide +kernel

theorem release_before_store_rejected :
    compile (originalTypedSource.map (mapTree reversePublication)) = none := by
  simp only [originalTypedSource, ↓List.map_cons, ↓List.map_nil, mapTree.eq_def, reversePublication]
  decide +kernel

/-- A supported source-expression change is actually propagated into the plan.
The compiler is not a hash check followed by returning a hard-coded source. -/
theorem source_expression_is_retained :
    compile (originalTypedSource.map (mapTree changeNext)) =
      some { originalPlan with nextInitial := .literal 7 } := by
  simp only [originalTypedSource, ↓List.map_cons, ↓List.map_nil, mapTree.eq_def, changeNext]
  decide +kernel

/-- Changing a generated expression changes execution. A source bridge cannot
silently discard a different next_pid expression and keep the old machine. -/
theorem changed_source_changes_execution :
    (decodePlan { originalPlan with nextInitial := .literal 7 } twoPrograms 0
      (initial fp32 zeroMemory)).2.phase = .scan 7 ∧
    (decodePlan originalPlan twoPrograms 0 (initial fp32 zeroMemory)).2.phase = .scan 1 := by
  decide +kernel

#axiomsClean changed_cas_condition_rejected
#axiomsClean missing_add_rejected
#axiomsClean ready_lock_replacement_rejected
#axiomsClean release_before_store_rejected
#axiomsClean source_expression_is_retained
#axiomsClean changed_source_changes_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.Frontend
