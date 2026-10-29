import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardInitial
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardMiddle

/-! Complete original and candidate backward execution and independent initial
snapshot gradients. Both actual source loops, the barrier, all rebases and all
pointer updates execute. Successful execution is a conclusion, including T=0.
The contract matches this JIT's actual d_q/d_k/d_v outputs; it does not assert
an unused final-state cotangent or an initial-state gradient. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardReal
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

structure Disjoint (r : BackwardRegions) extends BackwardSecondLoop.Disjoint r : Prop where
  q_dq : r.q ≠ r.dq
  k_dq : r.k ≠ r.dq
  v_dq : r.v ≠ r.dq
  do_dq : r.do_ ≠ r.dq

structure Post (p : Layout) (r : BackwardRegions) (hasSeed : Bool)
    (initial final : BlockState) : Prop where
  dq : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
    final.mem r.dq (keyGradientAddress p (launch initial) t k.val) =
      MemCell.real (MaskedMath.dq p (launch initial) initial r hasSeed t k)
  dk : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
    final.mem r.dk (keyGradientAddress p (launch initial) t k.val) =
      MemCell.real (MaskedMath.dk p (launch initial) initial r t k)
  dv : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
    final.mem r.dv (outputAddress p (launch initial) t j.val) =
      MemCell.real (MaskedMath.dv p (launch initial) initial r t j)
  frame : ∀ (region : RegionName) (off : Nat),
    (∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
      (r.dq,keyGradientAddress p (launch initial) t k.val) ≠ (region,off)) →
    (∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
      (r.dk,keyGradientAddress p (launch initial) t k.val) ≠ (region,off)) →
    (∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
      (r.dv,outputAddress p (launch initial) t j.val) ≠ (region,off)) →
    final.mem region off = initial.mem region off

theorem source_correct (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (hd : Disjoint r) :
    ∃ final, exec (BackwardSource.kernel p r hasSeed swapped).toAlgKernel initial =
      some final ∧ Post p r hasSeed initial final := by
  let l := launch initial
  let entry := BackwardInitial.state p r hasSeed initial
  obtain ⟨u,hu,h⟩ := BackwardFirstLoop.executes_remaining p l initial r hasSeed
    hd.q_dq hd.k_dq hd.v_dq hd.do_dq p.T 0 entry (by simp)
    (BackwardInitial.invariant p r hasSeed initial)
  have hfirst : stepStmt (.forRange "i" 0 p.T 1 (BackwardSource.firstBody p)) entry = some u := by
    rw [stepForRangeAux.forRange_unfold]
    exact hu
  let middle := BackwardMiddle.state p l initial r u
  have hm := BackwardMiddle.invariant p l initial r hasSeed u h
  obtain ⟨final,hfinal,hpost⟩ := BackwardSecondLoop.executes_remaining p l initial r hasSeed
    hd.toDisjoint p.T 0 middle (by simp) hm
  have hsecond : stepStmt (.forRange "_t" 0 p.T 1 (BackwardSource.secondBody p)) middle = some final := by
    rw [stepForRangeAux.forRange_unfold]
    exact hfinal
  refine ⟨final,?_,?_,?_,?_,?_⟩
  · rw [exec,BackwardSource.source_body]
    simp only [List.append_assoc]
    rw [stepStmts.append_some (BackwardInitial.executes p r hasSeed swapped initial)]
    change stepStmts (.forRange "i" 0 p.T 1 (BackwardSource.firstBody p) ::
      (BackwardSource.middleCode p r hasSeed swapped ++
        [.forRange "_t" 0 p.T 1 (BackwardSource.secondBody p)])) entry = some final
    rw [stepStmts.cons_some hfirst,
      stepStmts.append_some (BackwardMiddle.executes p l initial r hasSeed swapped u h.ready),
      stepStmts.cons_some hsecond,stepStmts.nil]
  · exact hpost.dq
  · intro t k ht hk
    exact hpost.dk t k (by omega) ht hk
  · intro t j ht hj
    exact hpost.dv t j (by omega) ht hj
  · intro region off hq hk hv
    exact hpost.frame region off hq (fun t k _ ht ha => hk t k ht ha)
      (fun t j _ ht ha => hv t j ht ha)

#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardReal
