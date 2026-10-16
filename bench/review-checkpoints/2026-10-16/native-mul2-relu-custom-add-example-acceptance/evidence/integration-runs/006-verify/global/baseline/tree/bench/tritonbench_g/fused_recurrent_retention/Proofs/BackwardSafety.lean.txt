import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstSafety
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondSafety
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardBoundarySafety

/-! Whole original/candidate backward trace safety. Safety follows actual
producer states from both loops. The final clipped pointer subtractions need
a separate finite-memory execution bridge; global FlattenOk does not hold. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
open VeriTile.Triton BackwardSource
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem source_safe (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (hd : BackwardReal.Disjoint r) (bounds : RegionBounds)
    (hb : Bounds p (launch initial) r hasSeed bounds) :
    (kernel p r hasSeed swapped).toAlgKernel.TraceSafe bounds initial := by
  rw [Kernel.TraceSafe,source_body]
  simp only [List.append_assoc]
  apply TraceProofs.append_safe
  · exact initial_safe p r hasSeed swapped initial bounds hb
  · intro u hu
    rw [BackwardInitial.executes] at hu
    cases hu
    let entry := BackwardInitial.state p r hasSeed initial
    have h := BackwardInitial.invariant p r hasSeed initial
    apply Stmt.TraceSafeList.cons_intro
    · exact first_loop_safe p (launch initial) initial r hasSeed bounds hb hd entry h
    · intro v he
      obtain ⟨v',he',hv⟩ := BackwardFirstLoop.executes_remaining p (launch initial) initial r hasSeed
        hd.q_dq hd.k_dq hd.v_dq hd.do_dq p.T 0 entry (by simp) h
      rw [stepForRangeAux.forRange_unfold] at he
      rw [he] at he'
      cases he'
      change Stmt.TraceSafeList bounds
        (middleCode p r hasSeed swapped ++ [.forRange "_t" 0 p.T 1 (secondBody p)]) v
      apply TraceProofs.append_safe
      · exact middle_safe p r hasSeed swapped bounds v
      · intro middle hm
        rw [BackwardMiddle.executes p (launch initial) initial r hasSeed swapped v hv.ready] at hm
        cases hm
        apply Stmt.TraceSafeList.cons_intro
        · exact second_loop_safe p (launch initial) initial r hasSeed bounds hb hd _
            (BackwardMiddle.invariant p (launch initial) initial r hasSeed v hv)
        · intro _ _; exact Stmt.TraceSafeList.nil_intro

#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
