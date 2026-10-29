import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSafety
import VeriTile.Triton.Memory.FlattenTolerant

/-! The complete prefix before the reverse loop preserves full state under
finite allocation translation. This includes the actual header, optional seed,
first loop, barrier, rebases and zero carry. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatPrefix
open VeriTile.Triton BackwardSource
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def code (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) : List Stmt :=
  initialCode p r hasSeed swapped ++ [.forRange "i" 0 p.T 1 (firstBody p)] ++
    middleCode p r hasSeed swapped

theorem split (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    (kernel p r hasSeed swapped).toAlgKernel.body =
      code p r hasSeed swapped ++ [.forRange "_t" 0 p.T 1 (secondBody p)] := by
  rw [source_body]
  simp only [code,List.append_assoc]

theorem flatten_ok (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool) :
    StmtList.FlattenOk (code p r hasSeed swapped) := by
  rw [code,BackwardHeaderSource.initial_exact,BackwardMiddleSource.middle_exact]
  cases swapped <;> simp [BackwardHeaderSource.header,BackwardHeaderSource.seedCode,
    BackwardHeaderSource.gradientPointerOp,ForwardHeaderSource.mulRef,
    ForwardHeaderSource.decayOp,ForwardHeaderSource.pointerOp,ForwardHeaderSource.keyMask,
    ForwardHeaderSource.valueMask,BackwardHeaderSource.maskKV,BackwardHeaderSource.statePointerOp,
    BackwardHeaderSource.seedAdd,BackwardHeaderSource.seedLoad,BackwardMiddleSource.code,
    BackwardMiddleSource.pointerOp,BackwardMiddleSource.gradientPointerOp,
    BackwardMiddleSource.outputPointerOp,BackwardMiddleSource.lastOffset,
    firstBody,ForwardSource.vectorLoad,stateUpdate,dqProduct,ForwardSource.advance,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,Op.FlattenOk.eq_def]

theorem executes (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (hd : BackwardReal.Disjoint r) :
    ∃ middle, stepStmts (code p r hasSeed swapped) initial = some middle ∧
      BackwardSecondLoop.Holds p (launch initial) initial r hasSeed p.T middle := by
  let l := launch initial
  let entry := BackwardInitial.state p r hasSeed initial
  obtain ⟨u,hu,h⟩ := BackwardFirstLoop.executes_remaining p l initial r hasSeed
    hd.q_dq hd.k_dq hd.v_dq hd.do_dq p.T 0 entry (by simp)
    (BackwardInitial.invariant p r hasSeed initial)
  have hfirst : stepStmt (.forRange "i" 0 p.T 1 (firstBody p)) entry = some u := by
    rw [stepForRangeAux.forRange_unfold]
    exact hu
  refine ⟨BackwardMiddle.state p l initial r u,?_,BackwardMiddle.invariant p l initial r hasSeed u h⟩
  rw [code,List.append_assoc,stepStmts.append_some (BackwardInitial.executes p r hasSeed swapped initial)]
  change stepStmts (.forRange "i" 0 p.T 1 (firstBody p) :: middleCode p r hasSeed swapped) entry = _
  rw [stepStmts.cons_some hfirst]
  exact BackwardMiddle.executes p l initial r hasSeed swapped u h.ready

theorem safe (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (hd : BackwardReal.Disjoint r) (bounds : RegionBounds)
    (hb : BackwardSafety.Bounds p (launch initial) r hasSeed bounds) :
    Stmt.TraceSafeList bounds (code p r hasSeed swapped) initial := by
  rw [code,List.append_assoc]
  apply TraceProofs.append_safe
  · exact BackwardSafety.initial_safe p r hasSeed swapped initial bounds hb
  · intro u hu
    rw [BackwardInitial.executes] at hu
    cases hu
    let entry := BackwardInitial.state p r hasSeed initial
    have h := BackwardInitial.invariant p r hasSeed initial
    apply Stmt.TraceSafeList.cons_intro
    · exact BackwardSafety.first_loop_safe p (launch initial) initial r hasSeed bounds hb hd entry h
    · intro v _
      exact BackwardSafety.middle_safe p r hasSeed swapped bounds v

#axiomsClean split
#axiomsClean flatten_ok
#axiomsClean executes
#axiomsClean safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFlatPrefix
