import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardLoopSafety
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardBoundarySafety

/-! The complete forward source is trace safe and belongs to the generic finite
allocation bridge's fragment. Neither claim assumes successful execution. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSafety
open VeriTile.Triton ForwardSource SafetyAccess
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem source_safe (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool)
    (initial : BlockState) (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (bounds : RegionBounds) (hb : Bounds p (launch initial) r hasSeed storeFinal bounds) :
    (kernel p r hasSeed storeFinal swapped).toAlgKernel.TraceSafe bounds initial := by
  rw [Kernel.TraceSafe,source_body,List.append_assoc]
  apply TraceProofs.append_safe
  · exact ForwardBoundarySafety.initial_safe p r hasSeed storeFinal swapped initial bounds hb
  · intro u hu
    rw [ForwardInitial.executes] at hu
    cases hu
    let entry := ForwardInitial.state p r hasSeed initial
    have h := ForwardInitial.invariant p r hasSeed initial
    apply Stmt.TraceSafeList.cons_intro
    · exact ForwardLoopSafety.loop_safe p (launch initial) initial r hasSeed storeFinal
        bounds hb hqo hko hvo entry h
    · intro v he
      obtain ⟨v',he',hv⟩ := ForwardLoop.executes_remaining p (launch initial) initial r hasSeed
        hqo hko hvo p.T 0 entry (by simp) h
      rw [stepForRangeAux.forRange_unfold] at he
      rw [he] at he'
      cases he'
      exact ForwardBoundarySafety.final_safe p (launch initial) initial r hasSeed storeFinal
        swapped bounds hb v hv.ready

theorem source_flatten_ok (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) :
    (kernel p r hasSeed storeFinal swapped).toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body,ForwardHeaderSource.initial_exact,ForwardHeaderSource.final_exact]
  cases swapped <;> simp [ForwardHeaderSource.header,ForwardHeaderSource.seedCode,
    ForwardHeaderSource.finalCode,ForwardHeaderSource.mulRef,ForwardHeaderSource.decayOp,
    ForwardHeaderSource.pointerOp,ForwardHeaderSource.outputPointerOp,ForwardHeaderSource.keyMask,
    ForwardHeaderSource.valueMask,ForwardHeaderSource.maskKV,ForwardHeaderSource.statePointerOp,
    StatePrimitives.seedAdd,StatePrimitives.seedLoad,loopBody,vectorLoad,stateUpdate,outputProduct,
    advance,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,Op.FlattenOk.eq_def]

#axiomsClean source_safe
#axiomsClean source_flatten_ok
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardSafety
