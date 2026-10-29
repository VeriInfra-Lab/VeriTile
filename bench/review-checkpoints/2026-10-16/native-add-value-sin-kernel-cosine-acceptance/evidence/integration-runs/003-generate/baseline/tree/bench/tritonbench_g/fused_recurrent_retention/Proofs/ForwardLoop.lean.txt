import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardStep
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Addressing

/-! Reached-state invariant for all forward time iterations. The output and
frame clauses concern complete tagged cells; only the three loop input regions
must stay separate from the output region. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardLoop
open VeriTile.Triton Primitives MaskedMath ForwardState
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

structure Holds (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState) : Prop where
  ready : Ready p l initial r hasSeed t u
  qmem : ∀ off, u.readMem r.q off = initial.readMem r.q off
  kmem : ∀ off, u.readMem r.k off = initial.readMem r.k off
  vmem : ∀ off, u.readMem r.v off = initial.readMem r.v off
  output : ∀ (k : Nat) (j : Fin p.BV), k < t → activeValue p l j.val →
    u.mem r.o (outputAddress p l k j.val) = MemCell.real (MaskedMath.output p l initial r hasSeed k j)
  frame : ∀ (region : RegionName) (off : Nat),
    (∀ (k : Nat) (j : Fin p.BV), k < t → activeValue p l j.val →
      (r.o,outputAddress p l k j.val) ≠ (region,off)) → u.mem region off = initial.mem region off

theorem iteration (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (h : Holds p l initial r hasSeed t u) :
    ∃ v, stepStmts (ForwardSource.loopBody p)
      (u.setReg "_t" .nat [] (Tile.scalar t)) = some v ∧
        Holds p l initial r hasSeed (t+1) v := by
  let entry := u.setReg "_t" .nat [] (Tile.scalar t)
  let pre := beforeStore p l initial r hasSeed t entry
  let v := result p l initial r hasSeed t entry
  have he := ForwardStep.executes p l initial r hasSeed t entry
    (ready_set_loop p l initial r hasSeed t t u h.ready)
    (by simpa [entry] using h.qmem) (by simpa [entry] using h.kmem)
    (by simpa [entry] using h.vmem)
  refine ⟨v,he,?_,?_,?_,?_,?_,?_⟩
  · exact ready_result p l initial r hasSeed t entry
      (ready_set_loop p l initial r hasSeed t t u h.ready)
  · intro off
    change (ForwardState.scatter p l initial r hasSeed t pre).readMem r.q off = _
    rw [ForwardState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hqo]
    simpa [pre,beforeStore,entry] using h.qmem off
  · intro off
    change (ForwardState.scatter p l initial r hasSeed t pre).readMem r.k off = _
    rw [ForwardState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hko]
    simpa [pre,beforeStore,entry] using h.kmem off
  · intro off
    change (ForwardState.scatter p l initial r hasSeed t pre).readMem r.v off = _
    rw [ForwardState.scatter,
      BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hvo]
    simpa [pre,beforeStore,entry] using h.vmem off
  · intro k j hkt hj
    change (ForwardState.scatter p l initial r hasSeed t pre).mem r.o
      (outputAddress p l k j.val) = _
    by_cases hkt' : k < t
    · rw [show (ForwardState.scatter p l initial r hasSeed t pre).mem r.o
        (outputAddress p l k j.val) = pre.mem r.o (outputAddress p l k j.val) from
        Memory.full_frame p.BV r.o (fun i => outputAddress p l t i.val)
          (MaskedMath.output p l initial r hasSeed t) (fun i => activeValue p l i.val)
          pre r.o (outputAddress p l k j.val) (by
            intro i hi heq
            have htk := (Addressing.output_unique p l t k i j hi hj
              (congrArg Prod.snd heq)).1
            omega)]
      simpa [pre,beforeStore,entry] using h.output k j hkt' hj
    · have hktEq : k = t := by omega
      subst k
      exact Memory.written_cell p.BV r.o (fun i => outputAddress p l t i.val)
        (MaskedMath.output p l initial r hasSeed t) (fun i => activeValue p l i.val)
        pre j hj (fun i hi heq => (Addressing.output_unique p l t t i j hi hj heq).2)
  · intro region off hn
    change (ForwardState.scatter p l initial r hasSeed t pre).mem region off = _
    rw [show (ForwardState.scatter p l initial r hasSeed t pre).mem region off =
        pre.mem region off from
      Memory.full_frame p.BV r.o (fun j => outputAddress p l t j.val)
        (MaskedMath.output p l initial r hasSeed t) (fun j => activeValue p l j.val)
        pre region off (fun j hj => hn t j (by omega) hj)]
    exact h.frame region off (fun k j hk hj => hn k j (by omega) hj)

theorem executes_remaining (p : Layout) (l : Launch) (initial : BlockState)
    (r : ForwardRegions) (hasSeed : Bool)
    (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (remaining : Nat) : ∀ (t : Nat) (u : BlockState), t + remaining = p.T →
      Holds p l initial r hasSeed t u →
      ∃ v, stepForRangeAux "_t" t p.T 1 (ForwardSource.loopBody p) u = some v ∧
        Holds p l initial r hasSeed p.T v := by
  induction remaining with
  | zero =>
      intro t u ht h
      have ht' : t = p.T := by omega
      subst t
      exact ⟨u,stepForRangeAux.step_ge (by decide) (by omega),h⟩
  | succ n ih =>
      intro t u ht h
      obtain ⟨v,hv,hr⟩ := iteration p l initial r hasSeed t u hqo hko hvo h
      obtain ⟨w,hw,hwReady⟩ := ih (t+1) v (by omega) hr
      refine ⟨w,?_,hwReady⟩
      rw [stepForRangeAux.step_lt (by decide) (by omega),hv]
      exact hw

#axiomsClean iteration
#axiomsClean executes_remaining
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardLoop
