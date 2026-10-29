import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondStep
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondMemory

/-! Reverse-loop induction over the remaining time rows. The previously
produced d_q and every untouched tagged cell are preserved; d_k and d_v refer
to independent suffix sums over the initial input snapshot. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondLoop
open VeriTile.Triton Primitives MaskedMath BackwardSecondState
noncomputable section
set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false

structure Disjoint (r : BackwardRegions) : Prop where
  q_dk : r.q ≠ r.dk
  q_dv : r.q ≠ r.dv
  k_dk : r.k ≠ r.dk
  k_dv : r.k ≠ r.dv
  v_dk : r.v ≠ r.dk
  v_dv : r.v ≠ r.dv
  do_dk : r.do_ ≠ r.dk
  do_dv : r.do_ ≠ r.dv
  dq_dk : r.dq ≠ r.dk
  dq_dv : r.dq ≠ r.dv
  dk_dv : r.dk ≠ r.dv

structure Holds (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (remaining : Nat) (u : BlockState) : Prop where
  ready : Ready p l initial r remaining u
  qmem : ∀ off, u.readMem r.q off = initial.readMem r.q off
  kmem : ∀ off, u.readMem r.k off = initial.readMem r.k off
  vmem : ∀ off, u.readMem r.v off = initial.readMem r.v off
  domem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off
  dq : ∀ (s : Nat) (k : Fin p.BK), s < p.T → activeKey p l k.val →
    u.mem r.dq (keyGradientAddress p l s k.val) = MemCell.real (MaskedMath.dq p l initial r hasSeed s k)
  dk : ∀ (s : Nat) (k : Fin p.BK), remaining ≤ s → s < p.T → activeKey p l k.val →
    u.mem r.dk (keyGradientAddress p l s k.val) = MemCell.real (MaskedMath.dk p l initial r s k)
  dv : ∀ (s : Nat) (j : Fin p.BV), remaining ≤ s → s < p.T → activeValue p l j.val →
    u.mem r.dv (outputAddress p l s j.val) = MemCell.real (MaskedMath.dv p l initial r s j)
  frame : ∀ (region : RegionName) (off : Nat),
    (∀ (s : Nat) (k : Fin p.BK), s < p.T → activeKey p l k.val →
      (r.dq,keyGradientAddress p l s k.val) ≠ (region,off)) →
    (∀ (s : Nat) (k : Fin p.BK), remaining ≤ s → s < p.T → activeKey p l k.val →
      (r.dk,keyGradientAddress p l s k.val) ≠ (region,off)) →
    (∀ (s : Nat) (j : Fin p.BV), remaining ≤ s → s < p.T → activeValue p l j.val →
      (r.dv,outputAddress p l s j.val) ≠ (region,off)) → u.mem region off = initial.mem region off

theorem iteration (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (t i : Nat) (ht : t < p.T) (u : BlockState)
    (hd : Disjoint r) (h : Holds p l initial r hasSeed (t+1) u) :
    ∃ v, stepStmts (BackwardSource.secondBody p)
      (u.setReg "_t" .nat [] (Tile.scalar i)) = some v ∧
        Holds p l initial r hasSeed t v := by
  let entry := u.setReg "_t" .nat [] (Tile.scalar i)
  let pre := beforeStores p l initial r t entry
  let v := result p l initial r t entry
  have hr := ready_set_loop p l initial r (t+1) i u h.ready
  have he := BackwardSecondStep.executes p l initial r t ht entry hr
    (by simpa [entry] using h.qmem) (by simpa [entry] using h.kmem)
    (by simpa [entry] using h.vmem) (by simpa [entry] using h.domem)
  refine ⟨v,he,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact ready_result p l initial r t entry hr
  · intro off
    change (BackwardSecondMemory.stores p l initial r t pre).readMem r.q off = _
    rw [BackwardSecondMemory.read_other _ _ _ _ _ _ _ _ hd.q_dk hd.q_dv]
    simpa [pre,beforeStores,entry] using h.qmem off
  · intro off
    change (BackwardSecondMemory.stores p l initial r t pre).readMem r.k off = _
    rw [BackwardSecondMemory.read_other _ _ _ _ _ _ _ _ hd.k_dk hd.k_dv]
    simpa [pre,beforeStores,entry] using h.kmem off
  · intro off
    change (BackwardSecondMemory.stores p l initial r t pre).readMem r.v off = _
    rw [BackwardSecondMemory.read_other _ _ _ _ _ _ _ _ hd.v_dk hd.v_dv]
    simpa [pre,beforeStores,entry] using h.vmem off
  · intro off
    change (BackwardSecondMemory.stores p l initial r t pre).readMem r.do_ off = _
    rw [BackwardSecondMemory.read_other _ _ _ _ _ _ _ _ hd.do_dk hd.do_dv]
    simpa [pre,beforeStores,entry] using h.domem off
  · intro s k hs hk
    change (BackwardSecondMemory.stores p l initial r t pre).mem r.dq
      (keyGradientAddress p l s k.val) = _
    rw [BackwardSecondMemory.full_frame _ _ _ _ _ _ _ _
      (by intro j hj heq; exact hd.dq_dk (congrArg Prod.fst heq).symm)
      (by intro j hj heq; exact hd.dq_dv (congrArg Prod.fst heq).symm)]
    simpa [pre,beforeStores,entry] using h.dq s k hs hk
  · intro s k hts hs hk
    change (BackwardSecondMemory.stores p l initial r t pre).mem r.dk
      (keyGradientAddress p l s k.val) = _
    by_cases heq : s = t
    · subst s
      exact BackwardSecondMemory.key_written p l initial r t pre hd.dk_dv k hk
    · rw [BackwardSecondMemory.key_future p l initial r t s pre hd.dk_dv (by omega) k hk]
      simpa [pre,beforeStores,entry] using h.dk s k (by omega) hs hk
  · intro s j hts hs hj
    change (BackwardSecondMemory.stores p l initial r t pre).mem r.dv
      (outputAddress p l s j.val) = _
    by_cases heq : s = t
    · subst s
      exact BackwardSecondMemory.value_written p l initial r t pre j hj
    · rw [BackwardSecondMemory.value_future p l initial r t s pre hd.dk_dv (by omega) j hj]
      simpa [pre,beforeStores,entry] using h.dv s j (by omega) hs hj
  · intro region off hnq hnk hnv
    change (BackwardSecondMemory.stores p l initial r t pre).mem region off = _
    rw [BackwardSecondMemory.full_frame _ _ _ _ _ _ _ _
      (fun k hk => hnk t k (by omega) ht hk) (fun j hj => hnv t j (by omega) ht hj)]
    exact h.frame region off hnq
      (fun s k hs hT hk => hnk s k (by omega) hT hk)
      (fun s j hs hT hj => hnv s j (by omega) hT hj)

theorem executes_remaining (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (hasSeed : Bool) (hd : Disjoint r) (remaining : Nat) :
    ∀ (cur : Nat) (u : BlockState), cur + remaining = p.T →
      Holds p l initial r hasSeed remaining u →
      ∃ v, stepForRangeAux "_t" cur p.T 1 (BackwardSource.secondBody p) u = some v ∧
        Holds p l initial r hasSeed 0 v := by
  induction remaining with
  | zero =>
      intro cur u ht h
      exact ⟨u,stepForRangeAux.step_ge (by decide) (by omega),h⟩
  | succ n ih =>
      intro cur u ht h
      obtain ⟨v,hv,hr⟩ := iteration p l initial r hasSeed n cur (by omega) u hd h
      obtain ⟨w,hw,hwReady⟩ := ih (cur+1) v (by omega) hr
      refine ⟨w,?_,hwReady⟩
      rw [stepForRangeAux.step_lt (by decide) (by omega),hv]
      exact hw

#axiomsClean iteration
#axiomsClean executes_remaining
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondLoop
