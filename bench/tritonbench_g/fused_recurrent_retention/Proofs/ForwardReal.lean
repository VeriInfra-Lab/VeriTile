import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardInitial
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardFinal

/-! Complete forward source correctness for original and candidate independently.
The specification refers only to the initial input snapshot and standalone
finite sums. Successful execution and the full tagged memory frame are results,
not hypotheses. This is the erased mathematical interpreter, not native IEEE. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

structure Post (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (initial final : BlockState) : Prop where
  output : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
    final.mem r.o (outputAddress p (launch initial) t j.val) =
      MemCell.real (MaskedMath.output p (launch initial) initial r hasSeed t j)
  finalState : storeFinal = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p (launch initial) k.val → activeValue p (launch initial) j.val →
    final.mem r.final (stateAddress p (launch initial) k.val j.val) =
      MemCell.real (MaskedMath.stateValue p (launch initial) initial r.k r.v r.initial hasSeed p.T k j)
  frame : ∀ (region : RegionName) (off : Nat),
    (∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
      (r.o,outputAddress p (launch initial) t j.val) ≠ (region,off)) →
    (storeFinal = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
      activeKey p (launch initial) k.val → activeValue p (launch initial) j.val →
      (r.final,stateAddress p (launch initial) k.val j.val) ≠ (region,off)) →
    final.mem region off = initial.mem region off

theorem source_correct (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool)
    (initial : BlockState) (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (hfinal : storeFinal = Bool.true → r.o ≠ r.final) :
    ∃ final, exec (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgKernel initial =
      some final ∧ Post p r hasSeed storeFinal initial final := by
  let entry := ForwardInitial.state p r hasSeed initial
  obtain ⟨u,hu,h⟩ := ForwardLoop.executes_remaining p (launch initial) initial r hasSeed
    hqo hko hvo p.T 0 entry (by simp) (ForwardInitial.invariant p r hasSeed initial)
  have hloop : stepStmt (.forRange "_t" 0 p.T 1 (ForwardSource.loopBody p)) entry = some u := by
    rw [stepForRangeAux.forRange_unfold]
    exact hu
  let final := ForwardFinal.result p (launch initial) initial r hasSeed storeFinal u
  refine ⟨final,?_,?_,?_,?_⟩
  · rw [exec,ForwardSource.source_body]
    simp only [List.append_assoc]
    rw [stepStmts.append_some (ForwardInitial.executes p r hasSeed storeFinal swapped initial)]
    change stepStmts (.forRange "_t" 0 p.T 1 (ForwardSource.loopBody p) ::
      ForwardSource.finalCode p r hasSeed storeFinal swapped) entry = some final
    rw [stepStmts.cons_some hloop,ForwardFinal.executes p (launch initial) initial r
      hasSeed storeFinal swapped u h.ready]
  · intro t j ht hj
    rw [ForwardFinal.output_unchanged p (launch initial) initial r hasSeed storeFinal u hfinal]
    exact h.output t j ht hj
  · intro hf k j hk hj
    subst storeFinal
    exact ForwardFinal.final_cell p (launch initial) initial r hasSeed u k j hk hj
  · intro region off hout hstate
    rw [ForwardFinal.frame p (launch initial) initial r hasSeed storeFinal u region off hstate]
    exact h.frame region off hout

#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardReal
